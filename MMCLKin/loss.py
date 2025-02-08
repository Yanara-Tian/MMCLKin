import torch
import torch.nn as nn
import numpy as np


##------------------------------loss function-----------------------------
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        # https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289
        # important to add reduction='none' to keep per-batch-item loss
        # alpha=0.25, gamma=2 from https://www.cnblogs.com/qi-yuan-008/p/11992156.html
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, x, target):
        logcosh = x + torch.nn.functional.softplus(-2 * x) - np.log(2)
        return torch.mean(logcosh)


class MultiTargetCrossEntropy(torch.nn.Module):
    def __init__(self, C_dim=2):
        super(MultiTargetCrossEntropy, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=C_dim)
        self.nll_loss = torch.nn.NLLLoss()

    def forward(self, input, target):
        # input = input.view(target.shape[0], self.C, target.shape[1])
        '''
            input: shoud be `(N, T, C)` where `C = number of classes` and `T = number of targets`
            target: should be `(N, T)` where `T = number of targets`
        '''
        assert input.shape[0] == target.shape[0]
        assert input.shape[1] == target.shape[1]
        out = self.log_softmax(input)
        out = self.nll_loss(out, target)
        return out


class ChamferLoss(nn.Module):
    def __init__(self, input_channels, reduction="mean"):
        # TODO: repsect reduction rule
        super(ChamferLoss, self).__init__()
        self.input_channels = input_channels

    def forward(self, x, y):
        batch_size = x.shape[0]
        num_points = min(x.shape[1], y.shape[1])
        x = x[:, :num_points, : self.input_channels]
        y = y[:, :num_points, : self.input_channels]
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        indices = torch.arange(0, num_points)
        rx = xx[:, indices, indices].unsqueeze(1).expand([batch_size, num_points, num_points])
        ry = yy[:, indices, indices].unsqueeze(1).expand([batch_size, num_points, num_points])
        pdist = rx.transpose(2, 1) + ry - 2 * zz
        loss = torch.mean(pdist.min(1)[0]) + torch.mean(pdist.min(2)[0])
        return torch.mul(loss, 0.005)


def get_loss(loss_str):
    d = {
        'mse': torch.nn.MSELoss(),
        'mae': torch.nn.L1Loss(),
        'huber': torch.nn.SmoothL1Loss(),
        'smae': torch.nn.SmoothL1Loss(),
        'bce': torch.nn.BCELoss(),
        'bcen': torch.nn.BCELoss(reduction="none"),
        'bcel': torch.nn.BCEWithLogitsLoss(),
        'bceln': torch.nn.BCEWithLogitsLoss(reduction="none"),
        'mtce': MultiTargetCrossEntropy(),
        'logcosh':LogCoshLoss(),
        'kl': torch.nn.KLDivLoss(),
        'hinge': torch.nn.HingeEmbeddingLoss(),
        'nll': torch.nn.NLLLoss(),
        'ce': torch.nn.CrossEntropyLoss(),
        'focal': FocalLoss(alpha=0.25),
        'chamfer_huber': {'chamfer': ChamferLoss(input_channels=7),
                          'huber': torch.nn.SmoothL1Loss()},
        'chamfer_ce': {'chamfer': ChamferLoss(input_channels=7),
                       'ce': torch.nn.CrossEntropyLoss()}
    }
    if loss_str not in d.keys():
        raise ValueError('loss not found')
    return d[loss_str]
