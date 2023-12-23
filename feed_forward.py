#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ffn)
        self.linear_2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        """
        ffn=max(0,xw1+b1)w2+b2
        两层的全连接网络，线性层 => ReLU非线性激活函数 => 线性层
        (batch_size, seq_len, d_model)
        """
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        return x
