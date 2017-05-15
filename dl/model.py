
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class Module(nn.Module):
    """
    The overall Model
    """
    def __init__(self, opts):
        super(Module, self).__init__()
        self._opts = opts
        self.char_emb = nn.Embedding(self._opts.vocab_len, self._opts.emb_size, padding_idx=0)
        self.pos_emb = nn.Embedding(self._opts.pos_len, self._opts.emb_size, padding_idx=0)

        self.encoder = Encoder(self._opts)
        self.decoder = AttenDecoder(self._opts)
    
    def forward

class Encoder(nn.Module):
    """
    The encoder model for conll
    """
    def __init__(self, opts):
        super(Encoder, self).__init__()
        self._opts = opts

        
        self.encoder = nn.LSTM(self._opts.emb_size, self._opts.hidden_size, batch_first = True, bidirectional = True)

        self.fc_h = nn.Linear(2 * self._opts.hidden_size, self._opts.hidden_size)
        self.fc_c = nn.Linear(2 * self._opts.hidden_size, self._opts.hidden_size)

        self.fc_pos_1 = nn.Linear(self._opts.max_pos_len * self._opts.emb_size, self._opts.max_pos_len * self._opts.emb_size / 2)
        self.fc_pos_2 = nn.Linear(self._opts.max_pos_len * self._opts.emb_size / 2, self._opts.hidden_size)

    def forward(self, input, pos, hidden):
        """
        Forward pass
        """

        input = self.char_emb(input)
        pos = self.pos_emb(pos)

        print pos.size()
        print pos.view(self._opts.batch_size, -1).size()
        f1 = F.tanh(self.fc_pos_1(pos.view(self._opts.batch_size, -1)))
        f2 = F.tanh(self.fc_pos_2(f1))

        pos_c_state = f2.view(1, self._opts.batch_size, -1)
        add_pos_hidden = (hidden[0], torch.cat((pos_c_state, pos_c_state), 0) )

        output, state = self.encoder(input, add_pos_hidden)
        he = state[0]
        ce = state[1]

        print he.size()

        he_reduced = self.fc_h(he.view(self._opts.batch_size, -1))
        hc_reduced = self.fc_c(ce.view(self._opts.batch_size, -1))

        return output, (he_reduced.view(1,self._opts.batch_size,-1), hc_reduced.view(1,self._opts.batch_size,-1))

    def init_hidden(self):
        """
        Initial the hidden state
        """
        h0 = Variable(torch.zeros(2, self._opts.batch_size, self._opts.hidden_size))
        c0 = Variable(torch.zeros(2, self._opts.batch_size, self._opts.hidden_size))

        if self._opts.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

class AttenDecoder(nn.Module):
    """
    Decoder with attention
    """
    def __init__(self, opts):
        super(AttenDecoder, self).__init__()
        self._opts = opts

        self.char_emb = nn.Embedding(self._opts.vocab_len, self._opts.emb_size, padding_idx=0)
        self.decoder = nn.LSTMCell(self._opts.emb_size, self._opts.hidden_size)

        # w_h for hidden_states from encoder
        self.wh = nn.Conv2d(2 * self._opts.hidden_size, 1, self._opts.hidden_size, bias=False)
        # w_s for state output by the decoder
        self.ws = nn.Linear(self._opts.hidden_size, self._opts.hidden_size)

        self.V = nn.Linear(2 * self._opts.hidden_size, self._opts.hidden_size)
        self.V_prime = nn.Linear(self._opts.hidden_size, self._opts.vocab_len)

    def forward(self, input, encoder_cell, encoder_states, target):

        input = self.char_emb(input)


        return
