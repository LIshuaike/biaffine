# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CHAR_LSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, output_dim):
        super(CHAR_LSTM, self).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim,
                            output_dim // 2,
                            batch_first=True,
                            bidirectional=True)
        
    def forward(self, x):
        mask = x.ne(0)
        lens = mask.sum(dim=1)

        x = pack_padded_sequence(self.embedding(x), lens, True, False)
        x, (hidden, _) = self.lstm(x)
        hidden = torch.cat(torch.unbind(hidden), dim=-1)

        return hidden
