#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F
from multi_head import MultiHeadAttention
from feed_forward import FeedForward


class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ffn=2048):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  num_heads=num_heads,
                                                  d_ffn=d_ffn)
                                    for _ in range(num_layers)])

    def forward(self, x, mask):
        """
        embedding => Nx (Multi-Head Attention => Feed Forward)
        (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ffn=2048):
        super().__init__()
        self.multi_head = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model=d_model, d_ffn=d_ffn)

    def forward(self, x, mask):
        """
        Multi-Head Attention => Add & Norm => Feed Forward => Add & Norm
        """
        _x = self.multi_head(q=x, k=x, v=x, mask=mask)
        x = x + _x
        x = F.layer_norm(x, x.size())

        _x = self.ffn(x)
        x = x + _x
        x = F.layer_norm(x, x.size())
        return x
