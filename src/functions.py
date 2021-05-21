import torch


def get_forward_mask(max_sen_len):
    zero_mask = torch.zeros(max_sen_len, max_sen_len)
    for i in range(max_sen_len):
        zero_mask[i][: i + 1] = torch.FloatTensor([1])
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


# def eval_test(model, )
