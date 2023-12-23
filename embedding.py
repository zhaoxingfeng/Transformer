#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, padding_idx, max_seq_len, d_model):
        super().__init__()
        self.word_emb = WordEmbedding(vocab_size, padding_idx, d_model)
        self.pos_emb = PositionEmbedding(max_seq_len, d_model)

    def forward(self, x):
        """
        单词embedding+位置embedding
        (batch_size, seq_len, d_model)
        """
        pos_embed = self.pos_emb(x)
        word_embed = self.word_emb(x)
        embed = pos_embed + word_embed
        return embed


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def forward(self, x):
        """
        位置embedding
        (seq_len, d_model)
        """
        lines = []
        for pos in range(self.max_seq_len):
            line = [math.sin(pos/10000**(j/self.d_model)) if j % 2 == 0 \
                        else math.cos(pos/10000**((j-1)/self.d_model)) \
                            for j in range(self.d_model)]
            lines.append(line)
        embed = torch.tensor(lines)
        batch_size, seq_len = x.size()
        embed = embed[:seq_len, :]
        return embed


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, padding_idx, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=embedding_dim,
                                padding_idx=padding_idx)

    def forward(self, x):
        """
        单词embedding
        (batch_size, seq_len, embedding_dim)
        """
        embed = self.emb(x)
        return embed
