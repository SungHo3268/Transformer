import torch


def get_masking(out, gpu, cuda):
    seq = out.shape[-2]
    zero_mask = torch.zeros(seq, seq)
    if gpu:
        zero_mask = zero_mask.to(torch.device(f'cuda:{cuda}'))
    for i in range(seq):
        zero_mask[i][:i+1] = torch.FloatTensor([1])
    out *= zero_mask
    inf_mask = (out == 0)
    inf_mask = inf_mask.to(torch.int32) * -1e+07
    if gpu:
        inf_mask = inf_mask.to(torch.device(f'cuda:{cuda}'))
    return out + inf_mask
