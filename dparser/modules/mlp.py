# -*- encoding: utf-8 -*-

from dparser.modules.dropout import SharedDropout
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(MLP, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x