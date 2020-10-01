# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CHAR_LSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, output_dim):
        super(CHAR_LSTM, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim,
                            output_dim // 2,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
                Characters of all tokens.
                Each token holds no more than `fix_len` characters, and the excess is cut off directly.
        Returns:
            ~torch.Tensor:
                The embeddings of shape ``[batch_size, seq_len, n_out]`` derived from the characters.
        """
        mask = x.ne(0)
        # [batch_size, seq_len]
        lens = mask.sum(-1)
        char_mask = lens.gt(0)

        x = self.embedding(x[char_mask])
        x = pack_padded_sequence(x, lens[char_mask], True, False)
        x, (hidden, _) = self.lstm(x)
        # [n, fix_len, n_out]
        hidden = torch.cat(torch.unbind(hidden), dim=-1)
        # [batch_size, seq_len, n_out]
        embed = hidden.new_zeros(*lens.shape, self.output_dim)
        embed = embed.masked_scatter_(char_mask.unsqueeze(-1), hidden)

        return embed
