import numpy as np
import torch
from torch import nn

from arch.VariationalBottleneck import VariationalBottleneck
from arch.BayesianLayer import BayesLinear


class VBMLP(nn.Module):
    def __init__(self, width=1024):
        super().__init__()
        self.width = width
        self.num_classes = 10
        self.data_shape = (3, 32, 32)

        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(self.data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, self.num_classes)

        self.flag = 0

        self.vb = VariationalBottleneck(in_shape=(width,))
        self.eps = torch.randn(size=(1, 256))
        self.learned_eps = torch.nn.Parameter(torch.randn(size=(1, 256)))

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)

        # client randomly samples a feature
        if self.flag == 0:
            eps = None
        # the attacker has no knowledge of sampled feature, use a randomly generated one instead
        elif self.flag == 1:
            eps = self.eps
        # our proposed attack uses the jointly-optimized one instead
        else:
            eps = self.learned_eps

        x = self.vb(x, eps)
        x = self.l3(x)
        return x

    def loss(self):
        return self.vb.loss()

    def resample(self):
        self.eps = torch.randn(size=(1, 256))
        self.learned_eps.copy_(torch.randn(size=(1, 256)))
