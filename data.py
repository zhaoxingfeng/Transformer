#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构造数据集
"""
import os
import json
import random
import torch


class Vocab(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.vocab_stoi, self.vocab_itos = self.load_vocab()
    
    def load_vocab(self):
        """从huggingface标准tokenizer.json中解析token=>id映射"""
        with open(self.file_path, "r") as f:
            stoi = json.load(f)["model"]["vocab"]
            itos = dict(zip(stoi.values(), stoi.keys()))
            return stoi, itos

    def stoi(self, tokens):
        """token转id"""
        if isinstance(tokens, str):
            return self.vocab_stoi.get(tokens, self.vocab_stoi.get("[UNK]"))
        else:
            return [self.vocab_stoi.get(token, self.vocab_stoi.get("[UNK]")) for token in tokens]

    def itos(self, ids):
        """id转token"""
        if isinstance(ids, int):
            return self.vocab_itos[ids]
        else:
            return [self.vocab_itos[id] for id in ids]
    
    def __len__(self):
        return len(self.vocab_stoi)


class Multi30kDataset(torch.utils.data.Dataset):
    def __init__(self, source_vocab, target_vocab, source_path, target_path, max_seq_len, num_examples=100):
        """英语<=>德语翻译数据集"""
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_seq_len = max_seq_len
        self.num_examples = num_examples
        self.source_tokens = self.get_tokenize(source_vocab, source_path)
        self.target_tokens = self.get_tokenize(target_vocab, target_path)
        
    def get_tokenize(self, vocab, file_path):
        sencents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line) < 2:
                    continue
                # 使用空格替换不间断空格，使用小写字母替换大写字母
                line = line.strip("\n").replace('\u202f', ' ').replace('\xa0', ' ').lower()
                # 在单词和标点符号之间插入空格
                line = [' ' + char if i > 0 and self.no_space(char, line[i - 1]) else char
                    for i, char in enumerate(line)]
                line = "".join(line)
                sencents.append(vocab.stoi(line.split(" ")))
        return sencents[:self.num_examples]
    
    def __getitem__(self, indx):
        source = self.source_tokens[indx] + [self.source_vocab.stoi("[PAD]")] * \
            (self.max_seq_len - len(self.source_tokens[indx]))
        target = self.target_tokens[indx] + [self.target_vocab.stoi("[PAD]")] * \
            (self.max_seq_len - len(self.target_tokens[indx]))
        return (torch.tensor(source[:self.max_seq_len]), torch.tensor(target[:self.max_seq_len]))

    def no_space(self, char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    def __len__(self):
        return len(self.source_tokens)


class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, max_seq_len, file_path, num_examples=100):
        """该数据集是一个包含1亿词汇的英文词库数据，是从维基百科优质文章和标杆文章中提取得到的"""
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_mask_len = int(0.15 * max_seq_len)
        self.file_path = file_path
        self.num_examples = num_examples
        self.sentences = self.get_raw_sentence()
        self.pair_sentences = self.get_pair_sentences()
        
    def get_raw_sentence(self):
        """读取数据集"""
        sentences = []
        with open(self.file_path, "r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                # 过滤只有一句的行
                cnt = len(line.split(" . "))
                if cnt < 3:
                    continue
                sentences.append([x.strip(" ") for x in line.split(" . ")[:-1]])
        return sentences

    def get_pair_sentences(self):
        """生成句子对"""
        pair_sentences = []
        # 一行生成多个pair句子
        for idx, sentence in enumerate(self.sentences):
            for i in range(0, len(sentence) - 1):
                # 50%概率下一句为随机样本
                is_next = 1
                s_a = sentence[i]
                s_b = sentence[i+1]
                if random.random() < 0.5:
                    is_next = 0
                    rn = idx
                    while rn == idx:
                        rn = random.randint(0, len(self.sentences)-1)
                    tmp = self.sentences[rn]
                    s_b = tmp[random.randint(0, len(tmp)-1)]
                s_a = s_a.split(" ")
                s_b = s_b.split(" ")

                # input_ids、token_type_ids
                input_tokens = ["[CLS]"] + s_a + ["[SEP]"] + s_b
                input_tokens = input_tokens[:self.max_seq_len]
                token_type_ids = [0] * (2 + len(s_a)) + [1] * len(s_b) + [0] * (self.max_seq_len-2-len(s_a+s_b))
                token_type_ids = token_type_ids[:self.max_seq_len]

                # 15%的词被mask，其中80%被mask填充，10%被随机填充，10%填充值采用原值
                cnt_mask = int(0.15 * len(input_tokens))
                idx_mask = [x for x in random.sample(range(len(input_tokens)), cnt_mask+2) if x not in [0, len(s_a)+1]]
                idx_mask = sorted(idx_mask[:cnt_mask])
                # weight用于计算mlm损失，仅对mask的词赋权重=1，其他=0
                mlm_wieght = [1] * len(idx_mask) + [0] * self.max_mask_len
                mlm_wieght = mlm_wieght[:self.max_mask_len]

                y_mask = []
                for i in idx_mask:
                    y_mask.append(self.vocab.stoi(input_tokens[i]))
                    if random.random() < 0.8:
                        input_tokens[i] = "[MASK]"
                    else:
                        if random.random() < 0.5:
                            token_indx = random.randint(0, len(self.vocab)-1)
                            input_tokens[i] = self.vocab.itos(token_indx)
                
                # 被mask的词的indx和真值
                idx_mask = idx_mask + [0] * (self.max_mask_len - len(idx_mask))
                idx_mask = idx_mask[:self.max_mask_len]
                y_mask = y_mask + [0] * (self.max_mask_len - len(y_mask))
                y_mask = y_mask[:self.max_mask_len]

                # 词转id
                input_ids = self.vocab.stoi(input_tokens)
                input_ids += (self.max_seq_len - len(input_ids)) * [self.vocab.stoi("[PAD]")]
                input_ids = input_ids[:self.max_seq_len]
                pair_sentences.append([input_ids, token_type_ids, idx_mask, mlm_wieght, y_mask, is_next])
        return pair_sentences[:self.num_examples]

    def __getitem__(self, idx):
        dt = self.pair_sentences[idx]
        return (torch.tensor(dt[0]), torch.tensor(dt[1]), torch.tensor(dt[2]), \
                torch.tensor(dt[3]), torch.tensor(dt[4]), torch.tensor(dt[5]))
    
    def __len__(self):
        return len(self.pair_sentences)
