
import numpy as np
from ot.sliced import sliced_wasserstein_distance as SWD
from geomloss import SamplesLoss

import torch

from ipdb import set_trace as debug

def compute_metrics(pred, ref, constraint=None, sample_size=None, p=2, blur=.05, scaling=.95):
    """
        pred: (B1, D)
        ref: (B2, D)
    """

    sample_size = min(pred.shape[0], ref.shape[0]) if sample_size is None else sample_size
    pred = shuffle(pred, sample_size=sample_size)
    ref = shuffle(ref, sample_size=sample_size)

    metrics = {}
    metrics["SWD"] = SWD(pred, ref)
    metrics["MMD"] = MMD(pred, ref).cpu()
    metrics["WD"] = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling)(pred, ref)
    if constraint is not None:
        metrics["infeasible_ratio"] = infeasible_ratio(pred, constraint)

    return metrics

def shuffle(x, sample_size):
    """
        x: (B, D)
        ===
        return: (sample_size, D)
    """
    idx = np.random.choice(x.shape[0], sample_size, replace=False)
    return x[idx]

def infeasible_ratio(x, constraint):
    return sum(constraint.is_feasible(x)[0]==0) / x.shape[0]

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def MMD(source, target):
    # adopted from https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
    B = int(source.size()[0])
    kernels = guassian_kernel(source, target)
    XX = kernels[:B, :B]
    YY = kernels[B:, B:]
    XY = kernels[:B, B:]
    YX = kernels[B:, :B]
    loss = torch.mean(XX + YY - XY -YX)
    return loss
