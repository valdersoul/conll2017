
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

    def forward(self, input, pos, label):

        input = self.char_emb(input)
        pos = self.pos_emb(pos)
        label = self.char_emb(label)

        hidden = self.encoder.init_hidden()
        encoder_output, encoder_state = self.encoder(input, pos, hidden)
        encoder_output.contiguous()

        output = self.decoder(encoder_state, encoder_output, label)
        

        return output


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

        f1 = F.tanh(self.fc_pos_1(pos.view(self._opts.batch_size, -1)))
        f2 = F.tanh(self.fc_pos_2(f1))

        pos_c_state = f2.unsqueeze(0)
        add_pos_hidden = (hidden[0], torch.cat((pos_c_state, pos_c_state), 0) )

        output, state = self.encoder(input, add_pos_hidden)
        he = state[0]
        ce = state[1]

        he_reduced = self.fc_h(he.view(self._opts.batch_size, -1))
        ce_reduced = self.fc_c(ce.view(self._opts.batch_size, -1))

        return output, (he_reduced, ce_reduced)

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
        self.decoder = nn.LSTMCell(self._opts.emb_size, self._opts.hidden_size)

        self._atten_size = 2 * self._opts.hidden_size

        # w_h for hidden_states from encoder
        self.wh = nn.Conv2d(2 * self._opts.hidden_size, self._atten_size, 1, bias=False)
        # w_s for state output by the decoder
        self.ws = nn.Linear(self._opts.hidden_size * 2, self._atten_size)

        self.we = nn.Conv1d(self._atten_size, 1, 1, bias=False)

        self.V = nn.Linear(2 * self._opts.hidden_size, self._opts.hidden_size)
        self.V_prime = nn.Linear(self._opts.hidden_size + self._atten_size, self._opts.vocab_len)

    def forward(self, encoder_state, encoder_output, target):
        encoder_outputs = encoder_output
        encoder_outputs = encoder_outputs.view(self._opts.batch_size, self._atten_size, 1, -1)
        # w_h * h_i ---> batch_size x seq_len x hidden_size
        encoder_features = self.wh(encoder_outputs).view(self._opts.batch_size, -1, self._atten_size)

        hs = encoder_state[0]
        cs = encoder_state[1]

        outputs = []
        # target size is batch_size x seq_len x emb_size
        for i in xrange(target.size()[1]):
            #compute w_s * s_i ------> batch_size * 1 * hidden_size
            decoder_features = self.ws(torch.cat((hs, cs), 1)).unsqueeze(1)
            decoder_features = decoder_features.expand_as(encoder_features)

            #compute v * (w_h * h_i + w_s * s_i)
            combined_features = (encoder_features + decoder_features).view(self._opts.batch_size, self._atten_size, -1)
            e = self.we(F.tanh(combined_features)).squeeze()

            a = F.softmax(e).unsqueeze(1)

            h_star = torch.bmm(a, encoder_output).squeeze()
            hs, cs = self.decoder(target[:,i], (hs, cs))

            feature = torch.cat((h_star, hs), 1)
            output = F.log_softmax(self.V_prime(feature))
            outputs.append(output)

        return outputs
