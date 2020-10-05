# -*- encoding: utf-8 -*-

import torch
import unicodedata
from collections import Counter
# from pytorch_pretrained_bert import BertTokenizer
from transformers import AutoTokenizer


class Vocab():
    pad = '<PAD>'
    unk = '<UNK>'
    bos = '<BOS>'
    eos = '<EOS>'

    def __init__(self, bert_vocab, words, chars, tags, rels):
        self.pad_index = 0
        self.unk_index = 1
        self.bos_index = 2
        # self.eos_index = 3

        self.tokenizer = AutoTokenizer.from_pretrained(bert_vocab)

        self.words = [self.pad, self.unk, self.bos] + sorted(words)
        self.chars = [self.pad, self.unk, self.bos] + sorted(chars)
        self.tags = [self.bos] + sorted(tags)
        self.rels = [self.bos] + sorted(rels)

        self.wtoi = {w: i for i, w in enumerate(self.words)}
        self.ctoi = {c: i for i, c in enumerate(self.chars)}
        self.ttoi = {t: i for i, t in enumerate(self.tags)}
        self.rtoi = {r: i for i, r in enumerate(self.rels)}

        # ids of punctuation that appear in words
        self.puncts = sorted(i for word, i in self.wtoi.items()
                             if self.is_punctuation(word))

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_tags = len(self.tags)
        self.n_rels = len(self.rels)
        self.n_init = self.n_words

    def __repr__(self):
        info = f"{self.__class__.__name__}:\n"
        info += f"{self.n_words} words\n"
        info += f"{self.n_chars} chars\n"
        info += f"{self.n_tags} tags\n"
        info += f"{self.n_rels} rels\n"

        return info

    def word2id(self, sequence):
        return torch.tensor(
            [self.wtoi.get(word.lower(), self.unk_index) for word in sequence])

    def char2id(self, sequence, max_length=20):
        char_ids = torch.zeros(len(sequence), max_length, dtype=torch.long)
        for i, word in enumerate(sequence):
            ids = torch.tensor(
                [self.ctoi.get(c, self.unk_index) for c in word[:max_length]])
            char_ids[i, :len(ids)] = ids

        return char_ids

    def tag2id(self, sequence):
        return torch.tensor([self.ttoi.get(tag, 0) for tag in sequence])

    def rel2id(self, sequence):
        return torch.tensor([self.rtoi.get(rel, 0) for rel in sequence])

    def id2tag(self, ids):
        return [self.tags[i] for i in ids]

    def id2rel(self, ids):
        return [self.rels[i] for i in ids]

    def load_embedding(self, embedding, smooth=True):
        words = [word.lower() for word in embedding.tokens]
        # 把unk替换为我们指定的token
        if embedding.unk:
            words[embedding.unk_index] = self.unk

        self.extend(words)
        self.embedding = torch.zeros(self.n_words, embedding.dim)
        self.embedding[self.word2id(words)] = embedding.vectors

        if smooth:
            self.embedding /= torch.std(self.embedding)

    def extend(self, words):
        self.words += sorted(set(words).difference(self.wtoi))
        self.chars += sorted(set(''.join(words)).difference(self.ctoi))

        self.wtoi = {w: i for i, w in enumerate(self.words)}
        self.ctoi = {c: i for i, c in enumerate(self.chars)}

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

    def numericalize(self, corpus, training=True):
        subwords, starts = [], []
        for seq in corpus.words:
            seq = [self.tokenizer.tokenize(token) for token in seq]
            seq = [piece if piece else ['[PAD]'] for piece in seq]
            seq = [['[CLS]']] + seq + [['[SEP]']]
            lengths = [0] + [len(piece) for piece in seq]
            # flatten the word pieces
            subwords.append(sum(seq, []))
            # record the start position of all words
            starts.append(torch.tensor(lengths).cumsum(0)[:-2])
        subwords = [torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
                    for tokens in subwords]
        mask = [torch.ones(len(tokens)) for tokens in subwords]
        start_mask = [~mask[i].byte().index_fill_(0, starts[i], 0)
                      for i in range(len(mask))]
        berts = [(i, j, k) for i, j, k in zip(subwords, mask, start_mask)]
        words = [self.word2id([self.bos] + seq) for seq in corpus.words]
        chars = [self.char2id([self.bos] + seq) for seq in corpus.words]
        tags = [self.tag2id([self.bos] + seq) for seq in corpus.tags]

        if not training:
            return berts, words, chars, tags

        arcs = [torch.tensor([0] + seq) for seq in corpus.heads]
        rels = [self.rel2id([self.bos] + seq) for seq in corpus.rels]

        return berts, words, chars, tags, arcs, rels

    @classmethod
    def from_corpus(cls, bert_vocab, corpus, min_freq=1):
        # 需要去重
        words = Counter(word.lower() for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in corpus.words for char in ''.join(seq)})
        tags = list({tag for seq in corpus.tags for tag in seq})
        rels = list({rel for seq in corpus.rels for rel in seq})

        vocab = cls(bert_vocab, words, chars, tags, rels)

        return vocab

    @classmethod
    def is_punctuation(cls, word):
        return all(unicodedata.category(char).startswith('P') for char in word)
