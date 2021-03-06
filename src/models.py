import sys
import os
sys.path.append(os.getcwd())
from src.layers import *


class Encoder(nn.Module):
    def __init__(self, dropout, hidden_layer_num, d_model, d_ff, head_num, gpu, cuda):
        super(Encoder, self).__init__()
        self.sub_layers = nn.ModuleList()
        for _ in range(hidden_layer_num):
            self.sub_layers.append(Encoder_Sublayer(d_model, d_ff, head_num, dropout, gpu, cuda))

    def forward(self, src, enc_pad_mask):
        """
        :param src: the source_input = (batch_size, max_sen_len)
        :param enc_pad_mask:
        :return: out = (batch_size, max_sen_len, d_model)
        """
        for sub_layer in self.sub_layers:
            src = sub_layer(src, enc_pad_mask)                # out = (batch_size, max_sen_len, d_model)
        return src


class Decoder(nn.Module):
    def __init__(self, dropout, hidden_layer_num, d_model, d_ff, head_num, gpu, cuda):
        super(Decoder, self).__init__()
        self.sub_layers = nn.ModuleList()
        for _ in range(hidden_layer_num):
            self.sub_layers.append(Decoder_Sublayer(d_model, d_ff, head_num, dropout, gpu, cuda))

    def forward(self, tgt, hs, dec_combined_mask, dec_pad_mask):
        """
        :param tgt: the target input = (batch_size, seq_len(max_sen_len))
        :param hs: the output of the last Encoder layer.    hs = (batch_size, max_sen_len, d_model)
        :param dec_combined_mask:
        :param dec_pad_mask:
        :return: out = (batch_size, seq_len(max_sen_len), d_model)
        """
        for sub_layer in self.sub_layers:
            tgt = sub_layer(tgt, hs, dec_combined_mask, dec_pad_mask)            # out = (batch_size, seq_len, d_model)
        return tgt


class Transformer(nn.Module):
    def __init__(self, V, D, embed_weight, max_sen_len, dropout, hidden_layer_num, d_model, d_ff, head_num, gpu, cuda):
        """
        :param V: Vocabulary size
        :param D: embedding dimension size
        :param embed_weight: shared weight matrix with Encoder, Decoder, and linear layer before softmax.
        :param max_sen_len: the maximum length of the sentence
        :param dropout: the dropout ratio
        :param hidden_layer_num: the number of Encoder and Decoder layers
        :param d_model: output dimension
        :param d_ff: position wise feed forward network hidden dimension size
        :param head_num: the number of head in multi-head attention
        :param gpu: (bool) do you want to use gpu computation?
        :param cuda: (int) gpu number
        """
        super(Transformer, self).__init__()
        self.gpu = gpu
        self.cuda = cuda
        self.embed_weight = embed_weight
        self.input_layer = InputLayer(D, self.embed_weight, max_sen_len, dropout, gpu, cuda)
        self.encoder = Encoder(dropout, hidden_layer_num, d_model, d_ff, head_num, gpu, cuda)
        self.decoder = Decoder(dropout, hidden_layer_num, d_model, d_ff, head_num, gpu, cuda)
        self.fc = nn.Linear(d_model, V, bias=False)
        self.fc.weight_ = self.embed_weight.T

    def forward(self, src_input, tgt_input):
        enc_pad_mask = get_pad_mask(src_input, self.gpu, self.cuda)
        src = self.input_layer(src_input)
        hs = self.encoder(src, enc_pad_mask)            # hs = (batch_size, max_sen_len, d_model)
        dec_combined_mask = get_combined_mask(tgt_input, self.gpu, self.cuda)
        dec_pad_mask = get_pad_mask(src_input, self.gpu, self.cuda)
        tgt = self.input_layer(tgt_input)
        out = self.decoder(tgt, hs, dec_combined_mask, dec_pad_mask)       # out = (batch_size, seq_len, d_model)
        out = self.fc(out)                      # out = (batch_size, seq_len(max_sen_len), V)
        return out
