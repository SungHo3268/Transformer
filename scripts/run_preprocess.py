import pickle
import sys
import os
sys.path.append(os.getcwd())
from src.preprocess import *


"""
'train.en' is a raw string file one sentence per line.
'train.tok.en' is applied tokenizer which split the words with special characters such as #, ;, etc.
'train.tok.clean.en' is cleaned sentence list excepted strange sentence like specially long...
'train.tok.bpe.32000.en' is applied BPE (Subword encoder).
"""
data_dir = 'datasets/nmt_data/wmt14_de_en'
pre_dir = 'datasets/nmt_data/wmt14_de_en/preprocessed'
max_sen_len = 256


# # make dictionary from vocab file applied tokenizer, clean, and bpe.
# p_vocab_bpe_32000 = os.path.join(data_dir, 'vocab.bpe.32000')
# word_to_id, id_to_word = make_dict(p_vocab_bpe_32000)
# with open(os.path.join(pre_dir, 'dictionary.pkl'), 'wb') as fw:
#     pickle.dump((word_to_id, id_to_word), fw)
with open(os.path.join(pre_dir, 'dictionary.pkl'), 'rb') as fr:
    word_to_id, id_to_word = pickle.load(fr)


# # make source, target from dataset file applied tokenizer, clean, and bpe.
# p_train_en = os.path.join(data_dir, 'train.tok.clean.bpe.32000.en')
# p_train_de = os.path.join(data_dir, 'train.tok.clean.bpe.32000.de')
# train_en = data_load(p_train_en)
# train_de = data_load(p_train_de)
#
# # get source and target (input, output)
# source, source_len = make_source(train_en, word_to_id, max_sen_len, padding_idx=0)
# target_input, target_output, target_len = make_target(train_de, word_to_id, max_sen_len, padding_idx=0)
#
# # shuffle the raw data
# per = np.random.permutation(len(source_len))
# source = source[per]
# source_len = source_len[per]
# target_input = target_input[per]
# target_output = target_output[per]
# target_len = target_len[per]
# with open(os.path.join(pre_dir, f'source_all.pkl'), 'wb') as fw:
#     pickle.dump((source, source_len), fw, protocol=4)
# with open(os.path.join(pre_dir, f'target_all.pkl'), 'wb') as fw:
#     pickle.dump((target_input, target_output, target_len), fw, protocol=4)
#
#
# # split the whole dataset into small segments
# seg_size = len(source_len)//5 + 1
# for i in range(5):
#     with open(os.path.join(pre_dir, f'source_{i}.pkl'), 'wb') as fw:
#         pickle.dump((source[seg_size*i: seg_size*(i+1)], source_len[seg_size*i: seg_size*(i+1)]), fw)
#     with open(os.path.join(pre_dir, f'target_{i}.pkl'), 'wb') as fw:
#         pickle.dump((target_input[seg_size*i: seg_size*(i+1)], target_output[seg_size*i: seg_size*(i+1)],
#                      target_len[seg_size*i: seg_size*(i+1)]), fw)
with open(os.path.join(pre_dir, 'source_0.pkl'), 'rb') as fr:
    source, source_len = pickle.load(fr)
with open(os.path.join(pre_dir, 'target_0.pkl'), 'rb') as fr:
    target_input, target_output, target_len = pickle.load(fr)
