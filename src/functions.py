import torch


def get_mask(max_sen_len, gpu, cuda):
    zero_mask = torch.zeros(max_sen_len, max_sen_len)
    if gpu:
        zero_mask = zero_mask.to(torch.device(f'cuda:{cuda}'))
    for i in range(max_sen_len):
        zero_mask[i][:i + 1] = torch.FloatTensor([1])
    inf_mask = (zero_mask == 0)
    inf_mask = inf_mask.to(torch.int32) * -1e+07
    if gpu:
        inf_mask = inf_mask.to(torch.device(f'cuda:{cuda}'))
    return zero_mask, inf_mask


def apply_mask(x, zero_mask, inf_mask):
    x *= zero_mask
    x += inf_mask
    return x
