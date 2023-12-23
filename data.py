#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi30k数据处理
数据路径: ~/.cache/torch/text/datasets/Multi30k/
参考文档: https://pytorch.org/tutorials/beginner/translation_transformer.html
"""
import os
import torch
import torchtext


SRC_LANG = 'de'
TGT_LANG = 'en'
token_map = {}
vocab_map = {}
token_map[SRC_LANG] = torchtext.data.utils.get_tokenizer('spacy', language='de_core_news_sm')
token_map[TGT_LANG] = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

def yield_tokens(data_iter, lang):
    map = {SRC_LANG: 0, TGT_LANG: 1}
    for dt in data_iter:
        yield token_map[lang](dt[map[lang]])

for lang in [SRC_LANG, TGT_LANG]:
    train_iter = torchtext.datasets.Multi30k(split='train', language_pair=(SRC_LANG, TGT_LANG))
    vocab_map[lang] = torchtext.vocab.build_vocab_from_iterator(
            yield_tokens(train_iter, lang),
            min_freq=1,
            specials=special_symbols,
            special_first=True)
    vocab_map[lang].set_default_index(UNK_IDX)

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

# convert raw strings into tensors indices
text_transform = {}
for lang in [SRC_LANG, TGT_LANG]:
    text_transform[lang] = sequential_transforms(token_map[lang], vocab_map[lang], tensor_transform)

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANG](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANG](tgt_sample.rstrip("\n")))

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# 词表大小
SRC_VOCAB_SIZE = len(vocab_map[SRC_LANG])
TGT_VOCAB_SIZE = len(vocab_map[TGT_LANG])
train_iter, valid_iter, test_iter = torchtext.datasets.Multi30k(
        split=("train", "valid", "test"),
        language_pair=(SRC_LANG, TGT_LANG))

