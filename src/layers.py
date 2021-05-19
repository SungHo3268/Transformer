import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from src.functions import *


class PositionalEncoding(nn.Module):
    def __init__(self, max_sen_len, D, gpu, cuda):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = torch.zeros(max_sen_len, D)
        if gpu:
            self.pos_encoding = self.pos_encoding.to(torch.device(f'cuda:{cuda}'))
        for pos in range(max_sen_len):
            for i in range(D):
                exponent = pos / (10000**(2*i/D))
                exponent = torch.FloatTensor([exponent])
                if i % 2 == 0:
                    self.pos_encoding[pos][i] = torch.sin(exponent)
                else:
                    self.pos_encoding[pos][i] = torch.cos(exponent)

    def forward(self, x):
        """
        :param x: It is a output of the embedding layer. x = (batch_size, max_sen_len, embedding_dim)
        :return: x = (batch_size, max_sen_len, embedding_dim)
        """
        x += self.pos_encoding
        return x


class InputLayer(nn.Module):
    def __init__(self, D, embed_weight, max_sen_len, dropout, gpu, cuda):
        super(InputLayer, self).__init__()
        self.D = D
        self.embedding = nn.Embedding.from_pretrained(embed_weight, freeze=False, padding_idx=0)
        self.positional = PositionalEncoding(max_sen_len, D, gpu, cuda)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: source_input = (batch_size, max_sen_len)
        :return: x = (batch_size, max_sen_len, embed_dim)
        """
        x = self.embedding(x)       # x = (batch_size, max_sen_len, embed_dim)
        x *= np.sqrt(self.D)
        x = self.positional(x)
        x = self.dropout(x)
        return x


class ScaledDotProdAtt(nn.Module):
    def __init__(self, d, max_sen_len, mask, gpu, cuda):
        super(ScaledDotProdAtt, self).__init__()
        self.d = d
        self.mask = mask
        if self.mask:
            self.zero_mask, self.inf_mask = get_forward_mask(max_sen_len, gpu, cuda)
        self.att_zero_mask, self.att_inf_mask = get_att_mask(max_sen_len, gpu, cuda)
        self.gpu = gpu
        self.cuda = cuda

    def forward(self, q, k, v, sen_len):
        """
        :param q: (batch_size, head_num, seq_len(max_sen_len), d_k)
        :param k: (batch_size, head_num, seq_len(max_sen_len), d_k)
        :param v: (batch_size, head_num, seq_len(max_sen_len), d_v)
        :param sen_len: (batch_size, )
        :return:
        """
        att = torch.matmul(q, k.transpose(2, 3))        # att = (batch_size, head_num, seq_len, seq_len)
        att = att / (self.d**0.5)
        if self.mask:
            # att = apply_forward_mask(att, self.zero_mask, self.inf_mask)
            self.att_zero_mask *= self.zero_mask
            self.att_inf_mask += self.inf_mask
        att = apply_att_mask(att, sen_len, self.att_zero_mask, self.att_inf_mask)
        att = F.softmax(att, dim=-1)
        att = torch.matmul(att, v)                      # att = (batch_size, head_num, seq_len, d_v)
        return att


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_num, max_sen_len, dropout, gpu, cuda, mask):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.d = int(d_model / head_num)
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProdAtt(self.d, max_sen_len, mask, gpu, cuda)
        self.linear_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.residual = None
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear_o.weight)

    def forward(self, q, k, v, sen_len):
        """
        :param q: (batch_size, seq_len (max_sen_len), embed_dim)
        :param k: (batch_size, seq_len (max_sen_len), embed_dim)
        :param v: (batch_size, seq_len (max_sen_len), embed_dim)
        :param sen_len:
        :return: out = (batch_size, seq_len(max_sen_len), d_model)
        """
        batch_size, seq_len, _ = q.size()
        self.residual = q
        q = self.linear_q(q).view(batch_size, seq_len, self.head_num, self.d)
        k = self.linear_k(k).view(batch_size, seq_len, self.head_num, self.d)
        v = self.linear_v(v).view(batch_size, seq_len, self.head_num, self.d)

        q = q.transpose(1, 2)                       # q = (batch_size, head_num, seq_len (max_sen_len), d_q)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = self.attention(q, k, v, sen_len)        # att = (batch_size, head_num, seq_len (max_sen_len), d_v)
        att = att.transpose(1, 2)                   # att = (batch_size, seq_len(max_sen_len), head_num, d_v)
        att = att.contiguous().view(batch_size, seq_len, -1)     # att = (batch_size, seq_len(max_sen_len), d_model)
        out = self.linear_o(att)                      # out = (batch_size, seq_len(max_sen_len), d_model)
        out = self.dropout(out)
        out += self.residual
        out = self.layer_norm(out)
        return out


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.residual = None
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        """
        :param x: the output of the MultiHeadAttention layer. x= (batch_size, seq_len(max_sen_len), d_model)
        :return: out = (batch_size, seq_len(max_sen_len), d_model)
        """
        self.residual = x
        out = self.linear1(x)           # out = (batch_size, seq_len(max_sen_len), d_ff)
        out = self.relu(out)
        out = self.linear2(out)         # out = (batch_size, seq_len(max_sen_len), d_model)
        out += self.residual
        out = self.layer_norm(out)
        return out


class Encoder_Sublayer(nn.Module):
    def __init__(self, d_model, d_ff, head_num, max_sen_len, dropout, gpu, cuda):
        super(Encoder_Sublayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, head_num, max_sen_len, dropout, gpu, cuda, mask=False)
        self.pos_feed_forward = PositionwiseFFN(d_model, d_ff, dropout)

    def forward(self, x, s_len):
        """
        :param x: (the output of the embedding layer) or (the output of the previous encoder sublayer)
               x = (batch_size, max_sen_len, d_model)
        :param s_len:
        :return: out = (batch_size, seq_len(max_sen_len), d_model)
        """
        att = self.multi_head_attention(x, x, x, s_len)        # x = (batch_size, seq_len(max_sen_len), d_model)
        out = self.pos_feed_forward(att)                # out = (batch_size, seq_len(max_sen_len), d_model)
        return out


class Decoder_Sublayer(nn.Module):
    def __init__(self, d_model, d_ff, head_num, max_sen_len, dropout, gpu, cuda):
        super(Decoder_Sublayer, self).__init__()
        self.masked_multi_head_attention = \
            MultiHeadAttention(d_model, head_num, max_sen_len, dropout, gpu, cuda, mask=True)
        self.multi_head_attention = MultiHeadAttention(d_model, head_num, max_sen_len, dropout, gpu, cuda, mask=False)
        self.pos_feed_forward = PositionwiseFFN(d_model, d_ff, dropout)

    def forward(self, x, hs, s_len, t_len):
        """
        :param x: (the output of the embedding layer) or (the output of the previous encoder sublayer)
               x = (batch_size, seq_len(max_sen_len), d_model)
        :param hs: the output of the last Encoder layer.    hs = (batch_size, max_sen_len, d_model)
        :param s_len:
        :param t_len:
        :return: out = (batch_size, seq_len(max_sen_len), d_model)
        """
        att1 = self.masked_multi_head_attention(x, x, x, t_len)    # att1 = (batch_size, seq_len(max_sen_len), d_model)
        att2 = self.multi_head_attention(att1, hs, hs, s_len)      # att2 = (batch_size, seq_len(max_sen_len), d_model)
        out = self.pos_feed_forward(att2)                       # out = (batch_size, seq_len(max_sen_len), d_model)
        return out
