#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from transformer import Transformer
import data
import random
random.seed(666)
torch.manual_seed(666)
torch.set_num_threads(6)


# 模型超参
epochs = 10
batch_size = 128
max_seq_len = 256
learning_rate = 1e-5

# 加载数据
en_tokenizer_path = "./tokenizer/en_tokenizer.json"
de_tokenizer_path = "./tokenizer/de_tokenizer.json"
train_en_path, train_de_path = "./data/multi30k/train.en", "./data/multi30k/train.de"
valid_en_path, valid_de_path = "./data/multi30k/valid.en", "./data/multi30k/valid.de"
source_vocab = data.Vocab(en_tokenizer_path)
target_vocab = data.Vocab(de_tokenizer_path)
train_dataset = data.Multi30kDataset(source_vocab, target_vocab, train_en_path, \
        train_de_path, max_seq_len, num_examples=1000)
valid_dataset = data.Multi30kDataset(source_vocab, target_vocab, valid_en_path, \
        valid_de_path, max_seq_len, num_examples=1000)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=False)

model = Transformer(vocab_src_size=len(source_vocab),
                    vocab_tgt_size=len(target_vocab),
                    src_pad_idx=source_vocab.stoi("[PAD]"),
                    tgt_pad_idx=target_vocab.stoi("[PAD]"),
                    max_seq_len=max_seq_len,
                    d_model=512,
                    num_heads=8,
                    num_layers=6,
                    d_ffn=2048)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=target_vocab.stoi("[PAD]"))
print(sum([param.nelement() for param in model.parameters()]))

# 训练
for epoch in range(epochs):
    for i, batch in enumerate(train_dataloader):
        src = batch[0]
        tgt = batch[1][:, :-1]
        label = batch[1][:, 1:]

        pred = model(src, tgt)
        loss = loss_fn(pred.contiguous().view(-1, pred.size(-1)), label.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch: {0}, iter: {1}, loss: {2}".format(epoch, i, loss.item()))

# 评估
for j, batch in enumerate(test_dataloader):
    src = batch[0]
    tgt = batch[1][:, :-1]
    label = batch[1][:, 1:]
    pred = model(src, tgt)
    loss = loss_fn(pred.contiguous().view(-1, pred.size(-1)), label.contiguous().view(-1))
    print("valid: {0}, iter: {1}, loss: {2}".format(epoch, j, loss.item()))
