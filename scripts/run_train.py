import argparse
from distutils.util import strtobool as _bool
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import numpy as np
from tqdm.auto import tqdm
import pickle
import time
import sys
import os
sys.path.append(os.getcwd())
from src.models import Transformer
from src.criterion import LabelSmoothingLoss


############################ Argparse ############################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=int, default=5678)
parser.add_argument('--max_sen_len', type=int, default=128)
parser.add_argument('--max_epoch', type=int, default=18)        # 1epoch = about 5612 steps/    18 epoch = 100K steps
parser.add_argument('--step_batch', type=int, default=30)       # 780 sentences are about 25000 tokens = 1 step
parser.add_argument('--batch_size', type=int, default=26)       # step_batch * batch_size = about 780 sentences = 1 step
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--eval_interval', type=int, default=10)
parser.add_argument('--gpu', type=_bool, default=True)
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()

log_dir = f'log/tf_{args.step_batch}s_{args.batch_size}b_{args.max_sen_len}t'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f)


############################ Tensorboard ############################
tb_dir = os.path.join(log_dir, 'tb')
ckpt_dir = os.path.join(log_dir, 'ckpt')
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
tb_writer = SummaryWriter(tb_dir)


############################ Hyperparameter ############################
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

############################ InitNET ############################
# load dictionary
pre_dir = 'datasets/nmt_data/wmt14_de_en/preprocessed'
with open(os.path.join(pre_dir, 'dictionary.pkl'), 'rb') as fr:
    word_to_id, id_to_word = pickle.load(fr)
V = len(word_to_id)

embed_weight = nn.parameter.Parameter(torch.empty(V, embed_dim), requires_grad=True)
nn.init.normal_(embed_weight, mean=0, std=embed_dim**(-0.5))
model = Transformer(V, embed_dim, embed_weight, args.max_sen_len, dropout,
                    hidden_layer_num, d_model, d_ff, head_num, args.gpu, args.cuda)
criterion = LabelSmoothingLoss(label_smoothing, V, ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0, betas=(beta1, beta2), eps=epsilon)
device = None
if args.gpu:
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)


############################ Start Train ############################
stack = 0
step_num = 0
total_loss = 0
for epoch in range(args.max_epoch):
    # load the preprocessed dataset
    print('Loading input data...')
    with open(os.path.join(pre_dir, f'source_all.pkl'), 'rb') as fr:
        src_input, src_len = pickle.load(fr)
    with open(os.path.join(pre_dir, f'target_all.pkl'), 'rb') as fr:
        tgt_input, tgt_output, tgt_len = pickle.load(fr)

    print('Shuffling the data...')
    per = np.random.permutation(len(src_len))
    src_input = src_input[per]
    src_len = src_len[per]
    tgt_input = tgt_input[per]
    tgt_output = tgt_output[per]
    tgt_len = tgt_len[per]

    print('Loading to gpu...')
    src_input = torch.from_numpy(src_input).to(torch.int64)
    src_len = torch.from_numpy(src_len).to(torch.int64)
    tgt_input = torch.from_numpy(tgt_input).to(torch.int64)
    tgt_output = torch.from_numpy(tgt_output).to(torch.int64)
    tgt_len = torch.from_numpy(tgt_len).to(torch.int64)

    # make batch
    print('Making batch...')
    train = TensorDataset(src_input, src_len, tgt_input, tgt_output, tgt_len)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for src, s_len, tgt_in, tgt_out, t_len in tqdm(train_loader, total=len(train_loader),
                                                   desc=f'epoch: {epoch+1}/{args.max_epoch}',
                                                   bar_format='{l_bar}{bar:20}{r_bar}'):
        if args.gpu:
            src = src.to(device)
            # s_len = s_len.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            # t_len = t_len.to(device)
        with autocast():
            out = model(src, tgt_in, s_len, t_len)
            # t0 = time.time()
            loss = criterion(out.view(-1, V), tgt_out.view(-1))
            loss /= args.step_batch
        # t1 = time.time()
        # print("criterion: ", t1-t0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2.)
        # t2 = time.time()
        # print("backward: ", t2-t1)
        total_loss += loss.data
        stack += 1
        if stack % args.step_batch == 0:
            step_num += 1
            optimizer.param_groups[0]['lr'] = d_model ** (-0.5) * np.minimum(step_num ** (-0.5),
                                                                             step_num * (warmup_steps ** (-1.5)))
            optimizer.step()
            optimizer.zero_grad()
            tb_writer.add_scalar('loss/step', total_loss, step_num)
            tb_writer.add_scalar('lr/step', optimizer.param_groups[0]['lr'], step_num)
            total_loss = 0
            if stack % args.eval_interval == 0:
                #######
                result = torch.softmax(out[10], dim=-1)
                result = torch.max(result, dim=-1)[1]
                result = result.to(torch.device('cpu'))
                sen = []
                temp = ''
                for idx in result:
                    word = id_to_word[int(idx)]
                    if '@@' in word:
                        temp += word.replace('@@', '')
                        continue
                    if temp:
                        sen.append(temp)
                        temp = ''
                    sen.append(word)
                if temp:                # if the last word was included '@@' add to sentence.
                    sen.append(temp)
                print(' '.join(sen))
                #######

        else:
            continue
        tb_writer.flush()
    print('Saving the model...')
    torch.save(model.state_dict(), os.path.join(log_dir, 'ckpt/model.ckpt'))
    torch.save(optimizer.state_dict(), os.path.join(log_dir, 'ckpt/optimizer.ckpt'))
    print('Complete..!')
    print('\n')
