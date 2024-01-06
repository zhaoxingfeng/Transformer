#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考文献：
http://arxiv.org/abs/1810.04805
https://github.com/google-research/bert
https://github.com/d2l-ai/d2l-zh/tree/master/chapter_natural-language-processing-pretraining
https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert
https://blog.csdn.net/zhaohongfei_358/article/details/126892383
"""
import torch
from torch import nn
import torch.nn.functional as F
from encoder import Encoder as TransformerEncoder


class BertEmbedding(nn.Module):
    """编码器"""
    def __init__(self, vocab_size=8000, hidden_size=768, max_seq_len=512, type_vocab_size=2):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.token_type_embedding = nn.Embedding(type_vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.position_ids = torch.arange(max_seq_len).unsqueeze(0)
    
    def forward(self, input_ids, token_type_ids):
        embeddings = self.word_embedding(input_ids)
        embeddings += self.token_type_embedding(token_type_ids)
        embeddings += self.position_embedding(self.position_ids)
        embeddings = torch.layer_norm(embeddings, embeddings.size())
        return embeddings


class MaskedLM(nn.Module):
    """掩蔽语言模型任务"""
    def __init__(self, vocab_size=8000, hidden_size=768):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = F.layer_norm(x, x.size())
        x = self.linear_2(x)
        return x


class NextSentencePrediction(nn.Module):
    """下一句预测任务"""
    def __init__(self, hidden_size=768):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)
        return x


class BertModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size=8000, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                max_seq_len=512, type_vocab_size=2, intermediate_size=3072, padding_idx=1):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = BertEmbedding(vocab_size=vocab_size,
                                       hidden_size=hidden_size,
                                       max_seq_len=max_seq_len,
                                       type_vocab_size=type_vocab_size)
        self.encoder = TransformerEncoder(d_model=hidden_size,
                                          num_heads=num_attention_heads,
                                          num_layers=num_hidden_layers,
                                          d_ffn=intermediate_size)
        self.mlm = MaskedLM(vocab_size, hidden_size)
        self.nsp = NextSentencePrediction(hidden_size)

    def forward(self, input_ids, token_type_ids, pred_positions):
        mask = self.get_pad_mask(input_ids, input_ids, pad_idx=self.padding_idx)
        x = self.embedding(input_ids, token_type_ids)
        x = self.encoder(x, mask)
        
        # mlm
        mlm_y_pred = self.mlm(x)
        # 每个input_ids随机选择15%的词进行mask，仅预测这些词
        batch_size, num_mlm_015 = pred_positions.size(0), pred_positions.size(1)
        idx_015_flatten = torch.repeat_interleave(torch.arange(batch_size), num_mlm_015)
        mlm_y_pred = mlm_y_pred[idx_015_flatten, pred_positions.reshape(-1)]

        # nsp
        nsp_y_pred = self.nsp(x[:, 0, :])
        return x, mlm_y_pred, nsp_y_pred
    
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
