import numpy as np
from tqdm.auto import tqdm
import sys
import os
sys.path.append(os.getcwd())


def data_load(file):
    with open(file, 'r', encoding='utf8') as f:
        data = f.readlines()
    return data


def make_dict(file):
    vocab = data_load(file)
    word_to_id = {'<pad>': 0, '<s>': 1, '</s>': 2}
    id_to_word = {0: '<pad>', 1: '<s>', 2: '</s>'}
    for word in vocab:
        word_to_id[word[:-1]] = len(word_to_id)
        id_to_word[len(id_to_word)] = word[:-1]
    return word_to_id, id_to_word


def make_source(dataset, word_to_id, max_sen_len, padding_idx=None):
    source = []
    source_len = []
    for line in tqdm(dataset, desc='Making source...', bar_format='{l_bar}{bar:20}{r_bar}'):
        sentence = [word_to_id['<s>']]
        for word in line.split():
            sentence.append(word_to_id[word])
        sentence += [word_to_id['</s>']]
        sen_len = len(sentence)
        if sen_len > max_sen_len:
            sen_len = max_sen_len
            sentence = sentence[:max_sen_len]
        source_len.append(sen_len)
        if padding_idx is not None:
            pad = max_sen_len - sen_len
            sentence += [padding_idx] * pad
        source.append(sentence)
    if padding_idx is not None:
        source = np.array(source)
        source_len = np.array(source_len)
    return source, source_len


def make_target(dataset, word_to_id, max_sen_len, padding_idx=None):
    target_input = []
    target_output = []
    target_len = []
    for line in tqdm(dataset, desc='Making target...', bar_format='{l_bar}{bar:20}{r_bar}'):
        sentence = [word_to_id['<s>']]
        for word in line.split():
            sentence.append(word_to_id[word])
        sentence += [word_to_id['</s>']]
        sentence_in = sentence[:-1]
        sentence_out = sentence[1:]
        sen_len = len(sentence_in)
        if sen_len > max_sen_len:
            sen_len = max_sen_len
            sentence_in = sentence_in[:max_sen_len]
            sentence_out = sentence_out[:max_sen_len]
        target_len.append(sen_len)
        if padding_idx is not None:
            pad = max_sen_len - sen_len
            sentence_in += [padding_idx] * pad
            sentence_out += [padding_idx] * pad
        target_input.append(sentence_in)
        target_output.append(sentence_out)
    if padding_idx is not None:
        target_input = np.array(target_input)
        target_output = np.array(target_output)
        target_len = np.array(target_len)
    return target_input, target_output, target_len
