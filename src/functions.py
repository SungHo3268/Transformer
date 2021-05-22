import torch


def get_forward_mask(max_sen_len):
    zero_mask = torch.ones(max_sen_len, max_sen_len)
    inf_mask = (zero_mask == 0)
    inf_mask = inf_mask.to(torch.int64) * -1e+9
    return inf_mask


def get_att_mask(max_sen_len):
    att_zero_mask = torch.zeros(max_sen_len+1, max_sen_len, max_sen_len)
    for i in range(max_sen_len+1):
        att_zero_mask[i][:, : i] = torch.FloatTensor([1])
    att_inf_mask = (att_zero_mask == 0)
    att_inf_mask = att_inf_mask.to(torch.int64) * -1e+9
    return att_inf_mask


def apply_att_mask(att, sen_len, att_inf_mask, gpu, cuda):
    """
    :param att: (batch_size, head_num, seq_len, seq_len)
    :param sen_len: (batch_size, )
    :param att_inf_mask: (max_sen_len+1, seq_len(max_sen_len), seq_len(max_sen_len))
    :param gpu:
    :param cuda:
    :return:
    """
    inf_mask = att_inf_mask[sen_len].unsqueeze(1)
    if gpu:
        inf_mask = inf_mask.to(torch.device(f'cuda:{cuda}'))
    att += inf_mask
    return att


def get_ahead_mask(seq_len, gpu, cuda):
    zero_ahaed_mask = torch.ones(seq_len, seq_len)
    if gpu:
        zero_ahaed_mask = zero_ahaed_mask.to(torch.device(f'cuda:{cuda}'))
    zero_ahaed_mask = 1 - torch.tril(zero_ahaed_mask)
    return zero_ahaed_mask                  # zero_ahead_mask = (seq_len, seq_len)


def get_pad_mask(tensor, gpu, cuda):
    """
    tensor = (batch_size, seq_len) == target or source input
    """
    batch_size, seq_len = tensor.size()
    zero_pad_mask = torch.zeros_like(tensor)
    zero_pad_mask = torch.eq(tensor, zero_pad_mask)
    return zero_pad_mask.view(batch_size, 1, 1, seq_len)        # zero_pad_mask = (batch_size, 1, 1, seq_len)


def get_combined_mask(tensor, gpu, cuda):
    zero_ahead_mask = get_ahead_mask(tensor.shape[-1], gpu, cuda)       # (batch,
    zero_pad_mask = get_pad_mask(tensor, gpu, cuda)
    mask = torch.max(zero_pad_mask, zero_ahead_mask)        # mask = (batch_size, 1, seq_len, seq_len)
    return mask