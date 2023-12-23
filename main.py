#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from transformer import Transformer
from data import *
torch.set_num_threads(4)
torch.manual_seed(666)


# 模型超参
epochs = 10
batch_size = 128
max_seq_len = 256
d_model = 512
num_heads = 8
num_layers = 6
d_ffn = 2048
learning_rate = 1e-5

# 加载数据
train_dataloader = torch.utils.data.DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
valid_dataloader = torch.utils.data.DataLoader(valid_iter, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_iter, batch_size=batch_size, collate_fn=collate_fn)

model = Transformer(vocab_src_size=SRC_VOCAB_SIZE,
                    vocab_tgt_size=TGT_VOCAB_SIZE,
                    src_pad_idx=PAD_IDX,
                    tgt_pad_idx=PAD_IDX,
                    max_seq_len=max_seq_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    d_ffn=d_ffn)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=PAD_IDX)

# 训练
for epoch in range(epochs):
    for i, batch in enumerate(train_dataloader):
        # (seq_len, batch_size) => (batch_size, seq_len)
        src = batch[0].transpose(0, 1)
        tgt = batch[1].transpose(0, 1)[:, :-1]
        label = batch[1].transpose(0, 1)[:, 1:]

        pred = model(src, tgt)
        loss = loss_fn(pred.contiguous().view(-1, pred.size(-1)), label.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch: {0}, iter: {1}, loss: {2}".format(epoch, i, loss.item()))

# 评估
for j, batch in enumerate(test_dataloader):
    src = batch[0].transpose(0, 1)
    tgt = batch[1].transpose(0, 1)[:, :-1]
    label = batch[1].transpose(0, 1)[:, 1:]
    pred = model(src, tgt)
    loss = loss_fn(pred.contiguous().view(-1, pred.size(-1)), label.contiguous().view(-1))
    print("valid: {0}, iter: {1}, loss: {2}".format(epoch, j, loss.item()))
