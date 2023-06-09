import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.autograd import Variable

from meta_gnn.mine.models.layers import ConcatLayer, CustomSequential


import meta_gnn.mine.utils as utils

torch.autograd.set_detect_anomaly(True)

EPS = 1e-6

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device:", device)


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):

        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()

    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())

    t_log = EMALoss.apply(x, running_mean)

    return t_log, running_mean


class GaussianLayer(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std
        # self.device = device

    def forward(self, x):
        return x + self.std * torch.randn_like(x)


class StatisticsNetwork(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layers = nn.Sequential(
            GaussianLayer(std=0.3),
            nn.Linear(x_dim + z_dim, 512),
            nn.ELU(),
            GaussianLayer(std=0.5),
            nn.Linear(512, 512),
            nn.ELU(),
            GaussianLayer(std=0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.layers(x)


class MyStatisticsNetwork(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layers = nn.Sequential(
            GaussianLayer(std=0.3),
            nn.Linear(x_dim + z_dim, 64),
            nn.ELU(),
            GaussianLayer(std=0.5),
            nn.Linear(64, 64),
            nn.ELU(),
            GaussianLayer(std=0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x)


class Mine(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method

        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = CustomSequential(ConcatLayer(), *T)
            else:
                self.T = CustomSequential(ConcatLayer(), T)
        else:
            self.T = T

    def forward(self, x, z, bound = 400, z_marg=None, ma_rate=0.01, ma_et=1):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        catz = torch.cat((x, z), dim=-1)
        t = self.T(catz).mean()
        cat_marg = torch.cat((x, z_marg), dim=-1)
        t_marg = self.T(cat_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi


def get_est(x_dim, z_dim):

    beta = 1e-3
    num_gpus = 1 if device == 'cuda' else 0
    t = MyStatisticsNetwork(x_dim, z_dim)
    mi_estimator = Mine(t, loss='mine')

    return mi_estimator


class T(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layers = CustomSequential(ConcatLayer(), nn.Linear(x_dim + z_dim, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 1))

    def forward(self, x, z):
        return self.layers(x, z)
