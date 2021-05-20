import torch
import torch.nn as nn
import torch.nn.functional as F


# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, label_smoothing, classes, softmax_dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1. - label_smoothing
#         self.label_smoothing = label_smoothing
#         self.V = classes
#         self.softmax_dim = softmax_dim
#
#     def forward(self, pred, label):
#         """
#         :param pred: (batch_size * seq_len(max_sen_len), V)
#         :param label: (batch_size * seq_len(max_sen_len), )
#         :return:
#         """
#         assert pred.ndim == 2
#         pred = pred.log_softmax(dim=self.softmax_dim)
#         label = label.to(torch.int64)
#         with torch.no_grad():
#             true_dist = torch.full_like(pred, self.label_smoothing / (self.V-2))
#             true_dist.scatter_(1, label.data.unsqueeze(1), self.confidence)
#             # pad loss mask
#             mask = (label == 0).to(torch.int64)     # mask = (batch_size * seq_len(max_sen_len), )
#             true_dist *= mask.unsqueeze(1)
#         return torch.sum(-true_dist * pred) / sum(mask)


# in OpenMNT
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=0):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)    # word itself, and pad token
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))   # register buffer is not a parameter, but in state_dict.

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        mask = (target == self.ignore_index)
        model_prob.masked_fill_(mask.unsqueeze(1), 0)      # broadcasting
        pred = output.log_softmax(dim=-1)
        return torch.sum(-pred*model_prob) / sum(target != self.ignore_index)
