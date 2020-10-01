from dparser.metrics import AttachmentMethod

import torch
import torch.nn as nn
import torch.distributed as dist
from dparser.utils.parallel import is_master
from dparser.utils.parallel import DistributedDataParallel as DDP


class Model():
    def __init__(self, config, vocab, parser):

        self.config = config
        self.vocab = vocab
        self.parser = parser

    def train(self, loader):

        self.parser.train()

        for i, (words, chars, tags, arcs, rels) in enumerate(loader):
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.parser(words, chars)

            loss = self.parser.get_loss(s_arc, s_rel, arcs, rels, mask)
            loss = loss / self.config.update_steps
            loss.backward()

            if (i + 1) % self.config.update_steps == 0:
                nn.utils.clip_grad_norm_(self.parser.parameters(),
                                         self.config.clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, loader, punct=True, partial=False):
        self.parser.eval()

        loss, metirc = 0, AttachmentMethod()

        for words, chars, tags, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            mask[:, 0] = 0

            s_arc, s_rel = self.parser(words, chars)
            loss += self.parser.get_loss(s_arc,
                                         s_rel,
                                         arcs,
                                         rels,
                                         mask,
                                         partial=partial)
            pred_arcs, pred_rels = self.parser.decode(s_arc, s_rel, mask)

            if partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if specified
            if not punct:
                puncts = words.new_tensor(self.vocab.puncts)
                mask &= words.unsqueeze(-1).ne(puncts).all(-1)

            metirc(pred_arcs, pred_rels, arcs, rels, mask)
        loss /= len(loader)

        return loss, metirc

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()

        all_arcs, all_rels = [], []
        for words, chars, tags in loader:
            mask = words.ne(self.vocab.pad_index)
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_arc, s_rel = self.parser(words, chars)
            # s_arc, s_rel = s_arc[mask], s_rel[mask]
            pred_arcs, pred_rels = self.parser.decode(s_arc, s_rel, mask)

            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_arcs, all_rels
