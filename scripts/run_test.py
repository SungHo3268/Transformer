import argparse
from distutils.util import strtobool as _bool
import sacrebleu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
import pickle
import sys
import os
sys.path.append(os.getcwd())
from src.models import Transformer
from src.functions import decoding


############################ Argparse ############################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=int, default=5678)
parser.add_argument('--step_batch', type=int, default=26)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--max_sen_len', type=int, default=128)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--gpu', type=_bool, default=True)
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()

# log_dir = f'log/tf_{args.step_batch}s_{args.batch_size}b_{args.max_sen_len}t'
log_dir = 'log/tf_15s_52b_128t'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


############################ InitNet ############################
# load dictionary
pre_dir = 'datasets/nmt_data/wmt14_de_en/preprocessed'
with open(os.path.join(pre_dir, 'dictionary.pkl'), 'rb') as fr:
    word_to_id, id_to_word = pickle.load(fr)

# Hyperparameter
V = len(word_to_id)
embed_dim = 512
hidden_layer_num = 6
d_ff = 2048
d_model = 512
head_num = 8
warmup_steps = 4000
beta1 = 0.9
beta2 = 0.98
epsilon = 10**(-9)
dropout = 0.1
label_smoothing = 0.1
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)


############################ Load pretrained model ############################
# load the save model
print("Load the pretrained Transformer model.")
embed_weight = nn.parameter.Parameter(torch.empty(V, embed_dim), requires_grad=True)
model = Transformer(V, embed_dim, embed_weight, args.max_sen_len, dropout,
                    hidden_layer_num, d_model, d_ff, head_num, args.gpu, args.cuda)
model.load_state_dict(torch.load(os.path.join(log_dir, 'ckpt/model.ckpt'), map_location='cuda:0'))
model.eval()
device = None
if args.gpu:
    device = torch.device(f'cuda:{args.cuda}')
else:
    device = torch.device('cpu')
model = model.to(device)

############################ Input Data ############################
print("Load the test dataset.")
data_dir = 'datasets/nmt_data/preprocessed'
with open(os.path.join(pre_dir, 'test_source_all.pkl'), 'rb') as fr:
    test_src_input, _ = pickle.load(fr)
with open(os.path.join(pre_dir, 'test_target_all.pkl'), 'rb') as fr:
    _, test_tgt_output, _ = pickle.load(fr)

test_src_input = torch.from_numpy(test_src_input).to(torch.int64)
test_tgt_output = torch.from_numpy(test_tgt_output).to(torch.int64)

print("Make dataset to batch.")
test_dataset = TensorDataset(test_src_input, test_tgt_output)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


############################ Start Test ############################
result = torch.empty(len(test_loader), args.batch_size, args.max_sen_len)
cur = 0
tgt_label = []
for src_in, tgt_out in tqdm(test_loader, total=len(test_loader), bar_format='{l_bar}{bar:20}{r_bar}'):
    tgt_in = torch.zeros_like(tgt_out)
    tgt_in[:, 0] = torch.tensor([1])
    tgt_in = tgt_in.to(torch.int64)
    if args.gpu:
        src_in = src_in.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
    for i in range(args.max_sen_len):
        out = model(src_in, tgt_in)             # out = (batch_size, max_sen_len, vocab_size)
        pred = torch.max(out, dim=-1)[1]        # pred = (batch_size, max_sen_len)
        if i != (args.max_sen_len - 1):
            tgt_in[:, i+1] = pred[:, i]
        result[cur, :, i] = pred[:, i]
    cur += 1
    tgt_label.append(tgt_out)
result = result.to(torch.int64).view(-1, args.max_sen_len)

# Convert index to string representation by id_to_word dictionary.
output = decoding(result, id_to_word)
label = decoding(tgt_label, id_to_word)

# Evaluate the SACRE BLEU
bleu = sacrebleu.corpus_bleu(output, [label], force=True, lowercase=False)
print("sacre bleu: ", bleu.score)

test_dir = os.path.join(log_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
with open(os.path.join(test_dir, 'output.txt'), 'w', encoding='utf8') as fw:
    fw.writelines(output)
with open(os.path.join(test_dir, 'label.txt'), 'w', encoding='utf8') as fw:
    fw.writelines(label)
