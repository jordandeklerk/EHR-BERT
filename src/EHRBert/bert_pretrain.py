from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .bert_config import BertConfig
from .bert import BERT, PreTrainedBertModel

logger = logging.getLogger(__name__)


def freeze_afterwards(model):
    for p in model.parameters():
        p.requires_grad = False


class ClsHead(nn.Module):
    def __init__(self, config: BertConfig, voc_size):
        super(ClsHead, self).__init__()
        self.cls = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(
        ), nn.Linear(config.hidden_size, voc_size))

    def forward(self, input):
        return self.cls(input)


class SelfSupervisedHead(nn.Module):
    def __init__(self, config: BertConfig, dx_voc_size, proc_voc_size):
        super(SelfSupervisedHead, self).__init__()
        self.multi_cls = nn.ModuleList([ClsHead(config, dx_voc_size), ClsHead(
            config, dx_voc_size), ClsHead(config, proc_voc_size), ClsHead(config, proc_voc_size)])

    def forward(self, dx_inputs, proc_inputs):
        # inputs (B, hidden)
        # output logits
        return self.multi_cls[0](dx_inputs), self.multi_cls[1](proc_inputs), self.multi_cls[2](dx_inputs), self.multi_cls[3](proc_inputs)


class BERT_Pretrain(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, proc_voc=None):
        super(BERT_Pretrain, self).__init__(config)
        self.dx_voc_size = len(dx_voc.word2idx)
        self.proc_voc_size = len(proc_voc.word2idx)

        self.bert = BERT(config, dx_voc, proc_voc)
        self.cls = SelfSupervisedHead(config, self.dx_voc_size, self.proc_voc_size)

        self.apply(self.init_bert_weights)

    def forward(self, inputs, dx_labels=None, proc_labels=None):
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        _, dx_bert_pool = self.bert(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, proc_bert_pool = self.bert(inputs[:, 1, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))

        dx2dx, proc2dx, dx2proc, proc2proc = self.cls(dx_bert_pool, proc_bert_pool)
        # output logits
        if proc_labels is None or dx_labels is None:
            return F.sigmoid(dx2dx), F.sigmoid(proc2dx), F.sigmoid(dx2proc), F.sigmoid(proc2proc)
        else:
            loss = F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + \
                   F.binary_cross_entropy_with_logits(proc2dx, dx_labels) + \
                   F.binary_cross_entropy_with_logits(dx2proc, proc_labels) + \
                   F.binary_cross_entropy_with_logits(proc2proc, proc_labels)
            return loss, F.sigmoid(dx2dx), F.sigmoid(proc2dx), F.sigmoid(dx2proc), F.sigmoid(proc2proc)
