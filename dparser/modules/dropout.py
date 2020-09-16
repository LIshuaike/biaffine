# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn


class SharedDropout(nn.Module):
    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()
        # p 把元素置为0的概率
        self.p = p
        self.batch_first = batch_first

    # 打印类信息，可以比较一下不实现此函数的打印信息
    def extra_repr(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def forward(self, x):
        # self.training的默认值是True，此值继承自父类
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p).unsqueeze(1)
            else:
                mask = self.get_mask(x[0], self.p)
            x = x * mask

        return x

    @staticmethod
    def get_mask(x, p):
        # a.new_empty(), torch.empty()
        # 对于伯努利分布，以p,1-p的概率取1,0
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        # 缩放，与nn.Dropout保持一致，对非0值补偿
        mask = mask / (1 - p)

        return mask


class IndependentDropout(nn.Module):
    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items):
        if self.training:
            masks = [
                x.new_empty(x.shape[:2]).bernoulli_(1 - self.p) for x in items
            ]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [
                item * mask.unsqueeze(dim=-1)
                for item, mask in zip(items, masks)
            ]

        return items
