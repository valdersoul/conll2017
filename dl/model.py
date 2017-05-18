
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

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

        self.dropout = nn.Dropout(p=self._opts.dropout)

        init.uniform(self.char_emb.weight, -1 , 1)
        init.uniform(self.pos_emb.weight, -1 , 1)

    def forward(self, input, pos, label):
        batch_size = pos.size(0)

        input = self.dropout(self.char_emb(input))
        pos = self.dropout(self.pos_emb(pos))
        label = self.dropout(self.char_emb(label))

        hidden = self.encoder.init_hidden(batch_size)
        encoder_output, encoder_state, pos_feature = self.encoder(input, pos, hidden)
        encoder_output.contiguous()

        (hs, cs), output = self.decoder(encoder_state, encoder_output, label, pos_feature)

        return (hs, cs), output
    
    def encode_once(self, input, pos):
        batch_size = input.size(0)

        input = self.dropout(self.char_emb(input))
        pos = self.dropout(self.pos_emb(pos))

        hidden = self.encoder.init_hidden(batch_size)
        encoder_output, encoder_state = self.encoder(input, pos, hidden)
        encoder_output.contiguous()
        return encoder_output, encoder_state
    
    def decode_once(self, encoder_state, encoder_output, input, initial_state = False):
        input = self.dropout(self.char_emb(input))
        (hs, cs), output = self.decoder(encoder_state, encoder_output, input, initial_state)
        return (hs, cs), output

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

        init.uniform(self.fc_h.weight, -1, 1)
        init.uniform(self.fc_c.weight, -1, 1)
        init.uniform(self.fc_pos_1.weight, -1, 1)
        init.uniform(self.fc_pos_2.weight, -1, 1)

    def forward(self, input, pos, hidden):
        """
        Forward pass
        """

        batch_size = input.size(0)
        f1 = F.tanh(self.fc_pos_1(pos.view(batch_size, -1)))
        f2 = F.tanh(self.fc_pos_2(f1))

        pos_c_state = f2

        output, state = self.encoder(input, hidden)

        he = state[0]
        ce = state[1]

        he_reduced = self.fc_h(he.view(batch_size, -1))
        ce_reduced = self.fc_c(ce.view(batch_size, -1))

        return output, (he_reduced, ce_reduced), pos_c_state

    def init_hidden(self, batch_size):
        """
        Initial the hidden state
        """
        h0 = Variable(torch.zeros(2, batch_size, self._opts.hidden_size), requires_grad = False)
        c0 = Variable(torch.zeros(2, batch_size, self._opts.hidden_size), requires_grad = False)

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
        self.decoder = nn.LSTMCell(self._opts.emb_size, self._opts.hidden_size)

        self._atten_size = 2 * self._opts.hidden_size

        self.line_in = nn.Linear(self._opts.hidden_size, self._opts.hidden_size * 2)
        self.line_out = nn.Linear(self._atten_size + self._opts.hidden_size * 2, self._opts.vocab_len)

        init.kaiming_uniform(self.line_in.weight)
        init.kaiming_uniform(self.line_out.weight)

    def forward(self, encoder_state, encoder_output, target, pos_feature, initial_state = True):
        batch_size = target.size(0)

        hs = encoder_state[0]
        cs = encoder_state[1]

        outputs = []
        # target size is batch_size x seq_len x emb_size
        for i in xrange(target.size()[1]):
            #compute w_s * s_i ------> batch_size * 1 * hidden_size
            hs, cs = self.decoder(target[:, i], (hs, cs))
            
            targetT = self.line_in(hs).unsqueeze(2)

            attn = torch.bmm(encoder_output, targetT).squeeze(2)
            softmax_score = F.softmax(attn).unsqueeze(1)

            h_star = torch.bmm(softmax_score, encoder_output).squeeze(1)

            feature = torch.cat((h_star, hs, pos_feature), 1)

            output = F.log_softmax(self.line_out(feature))
            outputs.append(output)

        return (hs, cs), outputs
