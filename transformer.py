#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考文档：
https://arxiv.org/abs/1706.03762
https://github.com/hyunwoongko/transformer/tree/master
https://github.com/jayparks/transformer
https://github.com/facebookresearch/llama
"""
import math
import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size=8000, padding_idx=1, max_seq_len=256, d_model=512):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = self.get_pos_emb(max_seq_len, d_model)

    def forward(self, x):
        """
        单词embedding+位置embedding
        (batch_size, seq_len, d_model)
        """
        word_embed = self.word_emb(x)
        _, seq_len = x.size()
        pos_embed = self.pos_emb[:seq_len, :]
        return word_embed + pos_embed
    
    def get_pos_emb(self, max_seq_len, d_model):
        """
        位置embedding
        (max_seq_len, d_model)
        """
        lines = []
        for pos in range(max_seq_len):
            line = [math.sin(pos/10000**(j/d_model)) if j % 2 == 0 \
                        else math.cos(pos/10000**((j-1)/d_model)) \
                            for j in range(d_model)]
            lines.append(line)
        embed = torch.tensor(lines)
        return embed


class Transformer(nn.Module):
    def __init__(self, vocab_src_size=8000, vocab_tgt_size=8000, src_pad_idx=1, tgt_pad_idx=1,
                 max_seq_len=256, d_model=512, num_heads=8, num_layers=6, d_ffn=2048):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.encoder_emb = TransformerEmbedding(vocab_size=vocab_src_size,
                                                padding_idx=src_pad_idx,
                                                max_seq_len=max_seq_len,
                                                d_model=d_model)
        self.decoder_emb = TransformerEmbedding(vocab_size=vocab_tgt_size,
                                                padding_idx=tgt_pad_idx,
                                                max_seq_len=max_seq_len,
                                                d_model=d_model)
        self.encoder = Encoder(d_model=d_model,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               d_ffn=d_ffn)
        self.decoder = Decoder(d_model=d_model,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               d_ffn=d_ffn)
        self.linear = nn.Linear(d_model, vocab_tgt_size)

    def forward(self, src, tgt):
        """
        embedding => encoder => decoder => linear
        (batch_size, tgt_seq_len, vocab_tgt_size)
        """
        # mask
        # src, (batch_size, seq_len, seq_len)
        src_mask = self.get_pad_mask(src, src, self.src_pad_idx)
        # tgt, (batch_size, seq_len, seq_len)
        tgt_mask = self.get_pad_mask(tgt, tgt, self.tgt_pad_idx) + self.get_attention_mask(tgt)
        # src-tgt, (batch_size, tgt_seq_len, src_seq_len)
        src_tgt_mask = self.get_pad_mask(tgt, src, self.src_pad_idx)

        # embedding
        src = self.encoder_emb(src)
        tgt = self.decoder_emb(tgt)

        # encoder, decoder
        src_enc = self.encoder(src, src_mask)
        out = self.decoder(src_enc, tgt, tgt_mask, src_tgt_mask)
        out = self.linear(out)
        return out
    
    def get_pad_mask(self, q_seq, k_seq, pad_idx):
        """
        一个batch中不同长度的序列在padding到相同长度后，对padding部分的信息进行掩盖
        (batch_size, q_seq_len, k_seq_len)
        """
        batch_size, q_seq_len = q_seq.size()
        batch_size, k_seq_len = k_seq.size()
        mask = (q_seq == pad_idx).unsqueeze(2)
        mask = mask.expand(batch_size, q_seq_len, k_seq_len)
        return mask
    
    def get_attention_mask(self, seq):
        """
        对一个batch中所有样本采用对称矩阵来掩盖掉当前时刻之后所有位置的信息
        (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len = seq.size()
        mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1) == 1
        return mask
