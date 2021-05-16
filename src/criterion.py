import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, classes, softmax_dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1. - label_smoothing
        self.label_smoothing = label_smoothing
        self.V = classes
        self.softmax_dim = softmax_dim

    def forward(self, pred, label):
        """
        :param pred: (batch_size * seq_len(max_sen_len), V)
        :param label: (batch_size * seq_len(max_sen_len))
        :return:
        """
        assert pred.ndim == 2
        pred = pred.log_softmax(dim=self.softmax_dim)
        label = label.to(torch.int64)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.label_smoothing / (self.V-1))
            true_dist.scatter_(1, label.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.softmax_dim))
