import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from __future__ import division

class Encoder(nn.Module):
    def __init__(self, opts):
        super(Encoder, self).__init__()
        self._opts = opts
        
        self.char_emb = nn.Embedding(self._opts.vocab_len, self._opts.emb_size, padding_idx=0)
        self.pos_emb = nn.Embedding(self._opts.pos_len, self._opts.emb_size, padding_idx=0)
        self.encoder = nn.LSTM(self._opts.emb_size, self._opts.hidden_size, batch_first = True, bidirectional = True)

        self.fc_h = nn.Linear(2 * self._opts.hidden_size, self._opts.hidden_size)
        self.fc_c = nn.Linear(2 * self._opts.hidden_size, self._opts.hidden_size)
        
        self.fc_pos_1 = nn.Linear(self._ops.pos_max_len * self._opts.emb_size, self._ops.pos_max_len * self._opts.emb_size / 2)
        self.fc_pos_2 = nn.Linear(self._ops.pos_max_len * self._opts.emb_size / 2, self._opts.hidden_size)
    
    def forward(self, input):
        pos = input['pos']

        f1 = F.tanh(self.fc_pos_1(pos))
        f2 = F.tanh(self.fc_pos_2(f1))

        


