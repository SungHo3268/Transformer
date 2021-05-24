import torch


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
    if gpu:
        zero_pad_mask = zero_pad_mask.to(torch.device(f'cuda:{cuda}'))
    zero_pad_mask = torch.eq(tensor, zero_pad_mask)
    return zero_pad_mask.view(batch_size, 1, 1, seq_len)        # zero_pad_mask = (batch_size, 1, 1, seq_len)


def get_combined_mask(tensor, gpu, cuda):
    zero_ahead_mask = get_ahead_mask(tensor.shape[-1], gpu, cuda)       # (batch,
    zero_pad_mask = get_pad_mask(tensor, gpu, cuda)
    mask = torch.max(zero_pad_mask, zero_ahead_mask)        # mask = (batch_size, 1, seq_len, seq_len)
    return mask


def decoding(encoded, id_to_word):
    decoded = []
    if type(encoded) == torch.Tensor:
        for line in encoded:
            sen = []
            temp = ''
            for idx in line:
                word = id_to_word[int(idx)]
                if word == '</s>':
                    sen.append(temp)
                    temp = ''
                    sen.append('\n')
                    break
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
            if sen[-1] != '\n':
                sen.append('\n')
            sentence = ' '.join(sen)
            decoded.append(sentence)
        return decoded

    else:
        for batch in encoded:
            for line in batch:
                sen = []
                temp = ''
                for idx in line:
                    word = id_to_word[int(idx)]
                    if word == '</s>':
                        sen.append(temp)
                        temp = ''
                        sen.append('\n')
                        break
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
                if sen[-1] != '\n':
                    sen.append('\n')
                sentence = ' '.join(sen)
                decoded.append(sentence)
        return decoded
