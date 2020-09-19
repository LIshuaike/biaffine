# -*- encoding: utf-8 -*-

from dparser.modules.dropout import SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.modules.rnn import apply_permutation


# class BiLSTM(nn.Module):
#     r"""
#     BiLSTM is an variant of the vanilla bidirectional LSTM adopted by Biaffine Parser
#     with the only difference of the dropout strategy.
#     It drops nodes in the LSTM layers (input and recurrent connections)
#     and applies the same dropout mask at every recurrent timesteps.
#     APIs are roughly the same as :class:`~torch.nn.LSTM` except that we remove the ``bidirectional`` option
#     and only allows :class:`~torch.nn.utils.rnn.PackedSequence` as input.
#     References:
#         - Timothy Dozat and Christopher D. Manning. 2017.
#           `Deep Biaffine Attention for Neural Dependency Parsing`_.
#     Args:
#         input_size (int):
#             The number of expected features in the input.
#         hidden_size (int):
#             The number of features in the hidden state `h`.
#         num_layers (int):
#             The number of recurrent layers. Default: 1.
#         dropout (float):
#             If non-zero, introduces a :class:`SharedDropout` layer on the outputs of each LSTM layer except the last layer.
#             Default: 0.
#     .. _Deep Biaffine Attention for Neural Dependency Parsing:
#         https://openreview.net/forum?id=Hk95PK9le
#     """
#     def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
#         super().__init__()

#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout = dropout

#         self.f_cells = nn.ModuleList()
#         self.b_cells = nn.ModuleList()
#         for _ in range(self.num_layers):
#             self.f_cells.append(
#                 nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
#             self.b_cells.append(
#                 nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
#             input_size = hidden_size * 2

#         self.reset_parameters()

#     def __repr__(self):
#         s = f"{self.input_size}, {self.hidden_size}"
#         if self.num_layers > 1:
#             s += f", num_layers={self.num_layers}"
#         if self.dropout > 0:
#             s += f", dropout={self.dropout}"

#         return f"{self.__class__.__name__}({s})"

#     def reset_parameters(self):
#         for param in self.parameters():
#             # apply orthogonal_ to weight
#             if len(param.shape) > 1:
#                 nn.init.orthogonal_(param)
#             # apply zeros_ to bias
#             else:
#                 nn.init.zeros_(param)

#     def permute_hidden(self, hx, permutation):
#         if permutation is None:
#             return hx
#         h = apply_permutation(hx[0], permutation)
#         c = apply_permutation(hx[1], permutation)

#         return h, c

#     def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
#         hx_0 = hx_i = hx
#         hx_n, output = [], []
#         steps = reversed(range(len(x))) if reverse else range(len(x))
#         if self.training:
#             hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

#         for t in steps:
#             last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
#             if last_batch_size < batch_size:
#                 hx_i = [
#                     torch.cat((h, ih[last_batch_size:batch_size]))
#                     for h, ih in zip(hx_i, hx_0)
#                 ]
#             else:
#                 hx_n.append([h[batch_size:] for h in hx_i])
#                 hx_i = [h[:batch_size] for h in hx_i]
#             hx_i = [h for h in cell(x[t], hx_i)]
#             output.append(hx_i[0])
#             if self.training:
#                 hx_i[0] = hx_i[0] * hid_mask[:batch_size]
#         if reverse:
#             hx_n = hx_i
#             output.reverse()
#         else:
#             hx_n.append(hx_i)
#             hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
#         output = torch.cat(output)

#         return output, hx_n

#     def forward(self, sequence, hx=None):
#         r"""
#         Args:
#             sequence (~torch.nn.utils.rnn.PackedSequence):
#                 A packed variable length sequence.
#             hx (~torch.Tensor, ~torch.Tensor):
#                 A tuple composed of two tensors `h` and `c`.
#                 `h` of shape ``[num_layers*2, batch_size, hidden_size]`` contains the initial hidden state
#                 for each element in the batch.
#                 `c` of shape ``[num_layers*2, batch_size, hidden_size]`` contains the initial cell state
#                 for each element in the batch.
#                 If `hx` is not provided, both `h` and `c` default to zero.
#                 Default: ``None``.
#         Returns:
#             ~torch.nn.utils.rnn.PackedSequence, (~torch.Tensor, ~torch.Tensor):
#                 The first is a packed variable length sequence.
#                 The second is a tuple of tensors `h` and `c`.
#                 `h` of shape ``[num_layers*2, batch_size, hidden_size]`` contains the hidden state for `t = seq_len`.
#                 Like output, the layers can be separated using ``h.view(num_layers, 2, batch_size, hidden_size)``
#                 and similarly for c.
#                 `c` of shape ``[num_layers*2, batch_size, hidden_size]`` contains the cell state for `t = seq_len`.
#         """
#         x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
#         batch_size = batch_sizes[0]
#         h_n, c_n = [], []

#         if hx is None:
#             ih = x.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)
#             h, c = ih, ih
#         else:
#             h, c = self.permute_hidden(hx, sequence.sorted_indices)
#         h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
#         c = c.view(self.num_layers, 2, batch_size, self.hidden_size)

#         for i in range(self.num_layers):
#             x = torch.split(x, batch_sizes)
#             if self.training:
#                 mask = SharedDropout.get_mask(x[0], self.dropout)
#                 x = [i * mask[:len(i)] for i in x]
#             x_f, (h_f, c_f) = self.layer_forward(x=x,
#                                                  hx=(h[i, 0], c[i, 0]),
#                                                  cell=self.f_cells[i],
#                                                  batch_sizes=batch_sizes)
#             x_b, (h_b, c_b) = self.layer_forward(x=x,
#                                                  hx=(h[i, 1], c[i, 1]),
#                                                  cell=self.b_cells[i],
#                                                  batch_sizes=batch_sizes,
#                                                  reverse=True)
#             x = torch.cat((x_f, x_b), -1)
#             h_n.append(torch.stack((h_f, h_b)))
#             c_n.append(torch.stack((c_f, c_b)))
#         x = PackedSequence(x, sequence.batch_sizes, sequence.sorted_indices,
#                            sequence.unsorted_indices)
#         hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
#         hx = self.permute_hidden(hx, sequence.unsorted_indices)

#         return x, hx


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(
                nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            self.b_cells.append(
                nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            input_size = hidden_size * 2  #下一层的输入是本层的输出

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.input_size}, {self.hidden_size}"
        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        s += ')'

        return s

    def reset_parameters(self):
        for i in self.parameters():
            # apply orthogonal_ to weight
            if len(i.shape) > 1:
                nn.init.orthogonal_(i)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(i)

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        h, c = hx
        init_h, init_c = h, c
        output, seq_len = [], len(x)
        # 正向、反向LSTM
        steps = reversed(range(seq_len)) if reverse else range(seq_len)
        if self.training:
            hid_mask = SharedDropout.get_mask(h, self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(h), batch_sizes[t]
            if last_batch_size < batch_size:
                h = torch.cat((h, init_h[last_batch_size:batch_size]))
                c = torch.cat((c, init_c[last_batch_size:batch_size]))
            else:
                h = h[:batch_size]
                c = c[:batch_size]
            h, c = cell(input=x[t], hx=(h, c))
            output.append(h)
            if self.training:
                h = h * hid_mask[:batch_size]
        if reverse:
            output.reverse()
        output = torch.cat(output)

        return output

    def forward(self, sequence, hx=None):
        x = sequence.data
        batch_sizes = sequence.batch_sizes.tolist()
        max_batch_size = batch_sizes[0]

        if hx is None:
            init = x.new_zeros(max_batch_size, self.hidden_size)
            hx = (init, init)

        output = []
        for layer in range(self.num_layers):
            if self.training:
                mask = SharedDropout.get_mask(x[:max_batch_size], self.dropout)
                mask = torch.cat(
                    [mask[:batch_size] for batch_size in batch_sizes])
                x *= mask
            x = torch.split(x, batch_sizes)
            f_output = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.f_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=False)
            b_output = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.b_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=True)
            x = torch.cat([f_output, b_output], -1)
            output.append(PackedSequence(x, sequence.batch_sizes))

        return output
