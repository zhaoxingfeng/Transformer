#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = int(d_model / num_heads)
        self.multi_head = nn.ModuleList([Attention(d_model,self.d_k) for _ in range(num_heads)])
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        """
        计算多头注意力，Concat(Attention head1,...,Attention headn) => linear
        (batch_size, seq_len, d_model)
        """
        heads = []
        for layer in self.multi_head:
            heads.append(layer(q, k, v, mask))
        out = self.linear(torch.cat(heads, 2))
        return out


class Attention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.w_q = nn.Linear(d_model, d_k)
        self.w_k = nn.Linear(d_model, d_k)
        self.w_v = nn.Linear(d_model, d_k)

    def forward(self, q, k, v, mask=None):
        """
        计算注意力，MatMul => Scale => Mask => SoftMax => MatMul
        (batch_size, seq_len, d_k)
        """
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        x = q.matmul(k.transpose(1, 2))
        x = x / self.d_k ** 0.5

        if mask is not None:
            x = x.masked_fill(mask == False, -1e-6)

        x = F.softmax(x, dim=-1)
        x = x.matmul(v)
        return x
