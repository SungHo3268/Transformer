import sys
import os
sys.path.append(os.getcwd())
from src.layers import *
import time


class Encoder(nn.Module):
    def __init__(self, max_sen_len, dropout,
                 hidden_layer_num, d_model, d_ff, head_num, gpu, cuda):
        super(Encoder, self).__init__()
        self.sub_layers = nn.ModuleList()
        for _ in range(hidden_layer_num):
            self.sub_layers.append(Encoder_Sublayer(d_model, d_ff, head_num, max_sen_len, dropout, gpu, cuda))

    def forward(self, src, s_len):
        """
        :param src: the source_input = (batch_size, max_sen_len)
        :param s_len:
        :return: out = (batch_size, max_sen_len, d_model)
        """
        for sub_layer in self.sub_layers:
            src = sub_layer(src, s_len)                # out = (batch_size, max_sen_len, d_model)
        return src


class Decoder(nn.Module):
    def __init__(self, max_sen_len, dropout,
                 hidden_layer_num, d_model, d_ff, head_num, gpu, cuda):
        super(Decoder, self).__init__()
        self.sub_layers = nn.ModuleList()
        for _ in range(hidden_layer_num):
            self.sub_layers.append(Decoder_Sublayer(d_model, d_ff, head_num, max_sen_len, dropout, gpu, cuda))

    def forward(self, tgt, hs, s_len, t_len):
        """
        :param tgt: the target input = (batch_size, seq_len(max_sen_len))
        :param hs: the output of the last Encoder layer.    hs = (batch_size, max_sen_len, d_model)
        :param s_len:
        :param t_len:
        :return: out = (batch_size, seq_len(max_sen_len), d_model)
        """
        for sub_layer in self.sub_layers:
            tgt = sub_layer(tgt, hs, s_len, t_len)            # out = (batch_size, seq_len(max_sen_len), d_model)
        return tgt


class Transformer(nn.Module):
    def __init__(self, V, D, embed_weight, max_sen_len, dropout,
                 hidden_layer_num, d_model, d_ff, head_num, gpu, cuda):
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
        self.embed_weight = embed_weight
        self.input_layer = InputLayer(D, self.embed_weight, max_sen_len, dropout, gpu, cuda)
        self.encoder = Encoder(max_sen_len, dropout, hidden_layer_num, d_model, d_ff, head_num, gpu, cuda)
        self.decoder = Decoder(max_sen_len, dropout, hidden_layer_num, d_model, d_ff, head_num, gpu, cuda)
        self.fc = nn.Linear(d_model, V, bias=False)
        self.fc.weight_ = self.embed_weight.T

    def forward(self, src_input, tgt_input, s_len, t_len):
        # print("\n")
        # t1 = time.time()
        src = self.input_layer(src_input)
        hs = self.encoder(src, s_len)            # hs = (batch_size, max_sen_len, d_model)
        # t2 = time.time()
        # print("Encoder: ", t2 - t1)
        tgt = self.input_layer(tgt_input)
        out = self.decoder(tgt, hs, s_len, t_len)       # out = (batch_size, seq_len(max_sen_len), d_model)
        # t3 = time.time()
        # print("Decoder: ", t3-t2)
        out = self.fc(out)                      # out = (batch_size, seq_len(max_sen_len), V)
        # t4 = time.time()
        # print("Fully connected: ", t4-t3)
        return out
