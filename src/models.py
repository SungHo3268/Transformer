import sys
import os
sys.path.append(os.getcwd())
from src.layers import *
import time


class Encoder(nn.Module):
    def __init__(self, D, embed_weight, max_sen_len, dropout,
                 hidden_layer_num, d_model, d_ff, head_num, gpu, cuda):
        """"""
        super(Encoder, self).__init__()
        self.input_layer = InputLayer(D, embed_weight, max_sen_len, dropout, gpu, cuda)
        self.sub_layers = nn.ModuleList()
        for _ in range(hidden_layer_num):
            self.sub_layers.append(Encoder_Sublayer(d_model, d_ff, head_num, max_sen_len, dropout, gpu, cuda))

    def forward(self, x):
        """
        :param x: the source_input = (batch_size, max_sen_len)
        :return: out = (batch_size, max_sen_len, d_model)
        """
        out = self.input_layer(x)               # x = (batch_size, max_sen_len, embed_dim)
        for sub_layer in self.sub_layers:
            out = sub_layer(out)                # out = (batch_size, max_sen_len, d_model)
        return out


class Decoder(nn.Module):
    def __init__(self, D, embed_weight, max_sen_len, dropout,
                 hidden_layer_num, d_model, d_ff, head_num, gpu, cuda):
        """"""
        super(Decoder, self).__init__()
        self.input_layer = InputLayer(D, embed_weight, max_sen_len, dropout, gpu, cuda)
        self.sub_layers = nn.ModuleList()
        for _ in range(hidden_layer_num):
            self.sub_layers.append(Decoder_Sublayer(d_model, d_ff, head_num, max_sen_len, dropout, gpu, cuda))

    def forward(self, x, hs):
        """
        :param x: the target input = (batch_size, seq_len(max_sen_len))
        :param hs: the output of the last Encoder layer.    hs = (batch_size, max_sen_len, d_model)
        :return: out = (batch_size, seq_len(max_sen_len), d_model)
        """
        out = self.input_layer(x)               # x = (batch_size, seq_len(max_sen_len), embed_dim)
        for sub_layer in self.sub_layers:
            out = sub_layer(out, hs)            # out = (batch_size, seq_len(max_sen_len), d_model)
        return out


class Transformer(nn.Module):
    def __init__(self, V, D, embed_weight, max_sen_len, dropout,
                 hidden_layer_num, d_model, d_ff, head_num, gpu, cuda):
        """"""
        super(Transformer, self).__init__()
        self.encoder = Encoder(D, embed_weight, max_sen_len, dropout,
                               hidden_layer_num, d_model, d_ff, head_num, gpu, cuda)
        self.decoder = Decoder(D, embed_weight, max_sen_len, dropout,
                               hidden_layer_num, d_model, d_ff, head_num, gpu, cuda)
        self.fc = nn.Linear(d_model, V, bias=False)
        self.fc.weight_ = embed_weight.T

    def forward(self, src_input, tgt_input):
        hs = self.encoder(src_input)            # hs = (batch_size, max_sen_len, d_model)
        out = self.decoder(tgt_input, hs)       # out = (batch_size, seq_len(max_sen_len), d_model)
        out = self.fc(out)                      # out = (batch_size, seq_len(max_sen_len), V)
        return out
