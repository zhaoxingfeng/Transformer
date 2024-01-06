#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from bert import BertModel
import data
import random
random.seed(666)
torch.manual_seed(666)
torch.set_num_threads(6)


# 模型超参
epochs = 10
batch_size = 10
max_seq_len = 512
learning_rate = 1e-5

# 加载数据
tokenizer_path = "./tokenizer/en_tokenizer.json"
data_path = "./data/wikitext-2/wiki.train.tokens"
vocab = data.Vocab(tokenizer_path)
train_dataset = data.WikiDataset(vocab, max_seq_len, data_path, num_examples=1000)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
vocab_size = len(vocab)

model = BertModel(vocab_size=vocab_size,
                  hidden_size=768,
                  num_hidden_layers=12,
                  num_attention_heads=12,
                  max_seq_len=max_seq_len,
                  type_vocab_size=2,
                  intermediate_size=3072,
                  padding_idx=vocab.stoi("[PAD]"))

loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(sum([param.nelement() for param in model.parameters()]))

# 训练
for epoch in range(epochs):
    for input_ids, token_type_ids, pred_positions, mlm_weights, mlm_y, nsp_y in train_dataloader:
        _, mlm_y_pred, nsp_y_pred = model(input_ids, token_type_ids, pred_positions)

        # mlm，预测15%被mask的词，被mask的词才会计算损失
        mlm_loss = loss_fn(mlm_y_pred.contiguous(), mlm_y.contiguous().view(-1))
        mlm_loss *= mlm_weights.view(-1)
        mlm_loss = mlm_loss.sum() / mlm_weights.sum()
        # nsp，预测第二句是否为句子对
        nsp_loss = loss_fn(nsp_y_pred.contiguous(), nsp_y.contiguous().view(-1))
        nsp_loss = nsp_loss.sum() / len(nsp_y)
        # total
        loss = mlm_loss + nsp_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch: {0}, mlm_loss: {1}, nsp_loss: {2}".format(epoch, mlm_loss.item(), nsp_loss.item()))
