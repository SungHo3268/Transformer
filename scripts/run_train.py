import argparse
from distutils.util import strtobool as _bool
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp
import numpy as np
from tqdm.auto import tqdm
import pickle
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
parser.add_argument('--max_epoch', type=int, default=18)  # 1epoch = about 5612 steps/    18 epoch = 100K steps
parser.add_argument('--step_batch', type=int, default=26)  # 780 sentences are about 25000 tokens = 1 step
parser.add_argument('--batch_size', type=int, default=30)  # step_batch * batch_size = about 780 sentences = 1 step
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--gpu', type=_bool, default=True)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--restart', type=_bool, default=False)
parser.add_argument('--restart_epoch', type=int, default=15)
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
epsilon = 10 ** (-9)
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
nn.init.normal_(embed_weight, mean=0, std=embed_dim ** (-0.5))
model = Transformer(V, embed_dim, embed_weight, args.max_sen_len, dropout,
                    hidden_layer_num, d_model, d_ff, head_num, args.gpu, args.cuda)
criterion = LabelSmoothingLoss(label_smoothing, V, ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0, betas=(beta1, beta2), eps=epsilon)
scaler = amp.GradScaler()

if args.restart:
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'model.ckpt'),
                                     map_location=f'cuda:{args.cuda}' if args.gpu else 'cpu'))
    optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, 'optimizer.ckpt'),
                                         map_location=f'cuda:{args.cuda}' if args.gpu else 'cpu'))
    # scaler.load_state_dict(torch.load(os.path.join(ckpt_dir, 'scaler.ckpt'),
    #                                   map_location=f'cuda:{args.cuda}' if args.gpu else 'cpu'))

device = None
if args.gpu:
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)


############################ Start Train ############################
stack = int((4378047 // args.batch_size) * (args.restart_epoch-1)) if args.restart else 0
step_num = int(stack // args.step_batch) if args.restart else 0
total_loss = 0
start_epoch = args.restart_epoch - 1 if args.restart else 0
for epoch in range(start_epoch, args.max_epoch):
    # load the preprocessed dataset
    print('Loading input data...')
    with open(os.path.join(pre_dir, 'source_all.pkl'), 'rb') as fr:
        src_input, _ = pickle.load(fr)
    with open(os.path.join(pre_dir, 'target_all.pkl'), 'rb') as fr:
        tgt_input, tgt_output, _ = pickle.load(fr)

    print('Shuffling the data...')
    per = np.random.permutation(len(src_input))
    src_input = src_input[per]
    tgt_input = tgt_input[per]
    tgt_output = tgt_output[per]

    print('Converting numpy to torch...')
    src_input = torch.from_numpy(src_input).to(torch.int64)
    tgt_input = torch.from_numpy(tgt_input).to(torch.int64)
    tgt_output = torch.from_numpy(tgt_output).to(torch.int64)

    # make batch
    print('Making batch...')
    train = TensorDataset(src_input, tgt_input, tgt_output)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for src, tgt_in, tgt_out in tqdm(train_loader, total=len(train_loader), desc=f'epoch: {epoch + 1}/{args.max_epoch}',
                                     bar_format='{l_bar}{bar:20}{r_bar}'):
        if args.gpu:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
        with amp.autocast():
            out = model(src, tgt_in)
            loss = criterion(out.view(-1, V), tgt_out.view(-1))
            loss /= args.step_batch
        scaler.scale(loss).backward()
        total_loss += loss.data
        stack += 1
        if step_num == 1000000:
            break
        if stack % args.step_batch == 0:
            step_num += 1
            optimizer.param_groups[0]['lr'] = d_model ** (-0.5) * np.minimum(step_num ** (-0.5),
                                                                             step_num * (warmup_steps ** (-1.5)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            tb_writer.add_scalar('loss/step', total_loss, step_num)
            tb_writer.add_scalar('lr/step', optimizer.param_groups[0]['lr'], step_num)
            tb_writer.flush()
            total_loss = 0
            if stack % args.eval_interval == 0:
                #######
                print('\n')
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
                        temp += word
                        sen.append(temp)
                        temp = ''
                        continue
                    sen.append(word)
                if temp:  # if the last word was included '@@' add to sentence.
                    sen.append(temp)
                print(' '.join(sen))
                #######
        else:
            continue
    print('Saving the model...')
    torch.save(model.state_dict(), os.path.join(log_dir, 'ckpt/model.ckpt'))
    torch.save(optimizer.state_dict(), os.path.join(log_dir, 'ckpt/optimizer.ckpt'))
    torch.save(scaler.state_dict(), os.path.join(log_dir, 'ckpt/scaler.ckpt'))
    print('Complete..!')
    print('\n')
