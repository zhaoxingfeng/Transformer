#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F
from multi_head import MultiHeadAttention
from feed_forward import FeedForward


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ffn):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  num_heads=num_heads,
                                                  d_ffn=d_ffn)
                                    for _ in range(num_layers)])

    def forward(self, src, tgt, tgt_mask, src_tgt_mask):
        """
        embedding => Nx (Masked Multi-Head Attention => Multi-Head Attention => Feed Forward)
        (batch_size, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            tgt = layer(src, tgt, tgt_mask, src_tgt_mask)
        return tgt


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn):
        super().__init__()
        self.multi_head1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.multi_head2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model=d_model, d_ffn=d_ffn)
    
    def forward(self, src, tgt, tgt_mask, src_tgt_mask):
        """
        Masked Multi-Head Attention => Add & Norm => Multi-Head Attention => Add & Norm => Feed Forward => Add & Norm
        (batch_size, seq_len, d_model)
        """
        _tgt = self.multi_head1(q=tgt, k=tgt, v=tgt, mask=tgt_mask)
        _tgt = tgt + _tgt
        _tgt = F.layer_norm(_tgt, _tgt.size())

        src_tgt = self.multi_head2(q=_tgt, k=src, v=src, mask=src_tgt_mask)
        src_tgt = _tgt + src_tgt
        src_tgt = F.layer_norm(src_tgt, src_tgt.size())


        out = self.ffn(src_tgt)
        out = out + src_tgt
        out = F.layer_norm(out, out.size())

        return out
