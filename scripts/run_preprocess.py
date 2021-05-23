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
max_sen_len = 128


# # make dictionary from vocab file applied tokenizer, clean, and bpe.
# p_vocab_bpe_32000 = os.path.join(data_dir, 'vocab.bpe.32000')
# word_to_id, id_to_word = make_dict(p_vocab_bpe_32000)
# with open(os.path.join(pre_dir, 'dictionary.pkl'), 'wb') as fw:
#     pickle.dump((word_to_id, id_to_word), fw)
with open(os.path.join(pre_dir, 'dictionary.pkl'), 'rb') as fr:
    word_to_id, id_to_word = pickle.load(fr)


# make source, target from dataset file applied tokenizer, clean, and bpe.
p_train_en = os.path.join(data_dir, 'train.tok.clean.bpe.32000.en')
p_train_de = os.path.join(data_dir, 'train.tok.clean.bpe.32000.de')
train_en = data_load(p_train_en)
train_de = data_load(p_train_de)

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


################################## Preprocess the test datasets ##################################
# p_test_en = os.path.join(data_dir, 'newstest2014.tok.clean.bpe.32000.en')
# p_test_de = os.path.join(data_dir, 'newstest2014.tok.clean.bpe.32000.de')
# test_en = data_load(p_test_en)
# test_de = data_load(p_test_de)
#
# test_src, test_src_len = make_source(test_en, word_to_id, max_sen_len, padding_idx=0)
# test_tgt_input, test_tgt_output, test_tgt_len = make_target(test_de, word_to_id, max_sen_len, padding_idx=0)
#
# per = np.random.permutation(len(test_src))
# test_src = test_src[per]
# test_src_len = test_src_len[per]
# test_tgt_input = test_tgt_input[per]
# test_tgt_output = test_tgt_output[per]
# test_tgt_len = test_tgt_len[per]
# with open(os.path.join(pre_dir, f'test_source_all.pkl'), 'wb') as fw:
#     pickle.dump((test_src, test_src_len), fw, protocol=4)
# with open(os.path.join(pre_dir, f'test_target_all.pkl'), 'wb') as fw:
#     pickle.dump((test_tgt_input, test_tgt_output, test_tgt_len), fw, protocol=4)
