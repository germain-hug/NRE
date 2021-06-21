import math

import torch


def log(x: torch.Tensor):
    if isinstance(x, (float, int)):
        return math.log(float(x))
    return torch.log(x + torch.finfo(torch.float32).eps)


def entropy(log_cmaps: torch.tensor):
    N = len(log_cmaps)
    return -(log_cmaps.exp().view(N, -1) * log_cmaps.view(N, -1)).sum(dim=1)
