import sys
import os
sys.path.append(os.getcwd())
from src.layers import *
import time


class Encoder(nn.Module):
    def __init__(self, D, embed_weight, max_sen_len, dropout,
                 hidden_layer_num, d_model, d_ff, head_num, gpu, cuda):
        super(Encoder, self).__init__()
        self.input_layer = InputLayer(D, embed_weight, max_sen_len, dropout, gpu, cuda)
        self.sub_layers = nn.ModuleList()
        for _ in range(hidden_layer_num):
            self.sub_layers.append(Encoder_Sublayer(d_model, d_ff, head_num, max_sen_len, dropout, gpu, cuda))

    def forward(self, x, s_len):
        """
        :param x: the source_input = (batch_size, max_sen_len)
        :param s_len:
        :return: out = (batch_size, max_sen_len, d_model)
        """
        out = self.input_layer(x)               # x = (batch_size, max_sen_len, embed_dim)
        for sub_layer in self.sub_layers:
            out = sub_layer(out, s_len)                # out = (batch_size, max_sen_len, d_model)
        return out


class Decoder(nn.Module):
    def __init__(self, D, embed_weight, max_sen_len, dropout,
                 hidden_layer_num, d_model, d_ff, head_num, gpu, cuda):
        super(Decoder, self).__init__()
        self.input_layer = InputLayer(D, embed_weight, max_sen_len, dropout, gpu, cuda)
        self.sub_layers = nn.ModuleList()
        for _ in range(hidden_layer_num):
            self.sub_layers.append(Decoder_Sublayer(d_model, d_ff, head_num, max_sen_len, dropout, gpu, cuda))

    def forward(self, x, hs, s_len, t_len):
        """
        :param x: the target input = (batch_size, seq_len(max_sen_len))
        :param hs: the output of the last Encoder layer.    hs = (batch_size, max_sen_len, d_model)
        :param s_len:
        :param t_len:
        :return: out = (batch_size, seq_len(max_sen_len), d_model)
        """
        out = self.input_layer(x)               # x = (batch_size, seq_len(max_sen_len), embed_dim)
        for sub_layer in self.sub_layers:
            out = sub_layer(out, hs, s_len, t_len)            # out = (batch_size, seq_len(max_sen_len), d_model)
        return out


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
        self.encoder = Encoder(D, self.embed_weight, max_sen_len, dropout,
                               hidden_layer_num, d_model, d_ff, head_num, gpu, cuda)
        self.decoder = Decoder(D, self.embed_weight, max_sen_len, dropout,
                               hidden_layer_num, d_model, d_ff, head_num, gpu, cuda)
        self.fc = nn.Linear(d_model, V, bias=False)
        self.fc.weight_ = self.embed_weight.T

    def forward(self, src_input, tgt_input, s_len, t_len):
        print("\n")
        t1 = time.time()
        hs = self.encoder(src_input, s_len)            # hs = (batch_size, max_sen_len, d_model)
        t2 = time.time()
        print("Encoder: ", t2 - t1)
        out = self.decoder(tgt_input, hs, s_len, t_len)       # out = (batch_size, seq_len(max_sen_len), d_model)
        t3 = time.time()
        print("Decoder: ", t3-t2)
        out = self.fc(out)                      # out = (batch_size, seq_len(max_sen_len), V)
        t4 = time.time()
        print("Fully connected: ", t4-t3)
        return out
