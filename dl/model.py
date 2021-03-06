
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

    def forward(self, input, pos, label):

        encoder_state, encoder_output, pos_feature, input = self.encode_once(input, pos)

        label = self.dropout(self.char_emb(label))
        state, output = self.decoder(encoder_state, encoder_output, label, input, pos_feature, self.char_emb)

        return state, output

    def encode_once(self, input, pos):
        batch_size = pos.size(0)

        input = self.dropout(self.char_emb(input))
        pos = self.dropout(self.pos_emb(pos))

        hidden = self.encoder.init_hidden(batch_size)
        encoder_output, encoder_state, pos_feature = self.encoder(input, pos, hidden)
        encoder_output.contiguous()

        return encoder_state, encoder_output, pos_feature, input

    def decode_once(self, encoder_state, encoder_output, word, input, pos_feature, pre_state=None):

        word = self.char_emb(word)
        input = self.char_emb(input)

        state, output = self.decoder(encoder_state, encoder_output, word, input, pos_feature, self.char_emb, pre_state)
        return state, output

class Encoder(nn.Module):
    """
    The encoder model for conll
    """
    def __init__(self, opts):
        super(Encoder, self).__init__()
        self._opts = opts

        self.encoder = nn.GRU(2 * self._opts.emb_size, self._opts.hidden_size, batch_first=True, dropout=self._opts.dropout, bidirectional=True)

        self.fc_h = nn.Linear(2 * self._opts.hidden_size, self._opts.hidden_size)
        #self.fc_c = nn.Linear(2 * self._opts.hidden_size, self._opts.hidden_size)

        pos_size = self._opts.max_pos_len * self._opts.emb_size

        self.fc_pos_1 = nn.Linear(pos_size, pos_size / 2)
        self.fc_pos_2 = nn.Linear(pos_size / 2, self._opts.emb_size)

        self.dropout = nn.Dropout(p=self._opts.dropout)

        self.bn = nn.BatchNorm1d(2 * self._opts.emb_size)

        for w in self.encoder.all_weights:
            nn.init.orthogonal(w[0])

    def forward(self, input, pos, hidden):
        """
        Forward pass
        """

        batch_size = input.size(0)
        f1 = F.tanh(self.fc_pos_1(pos.view(batch_size, -1)))
        f2 = F.tanh(self.fc_pos_2(f1))
        pos_c_state = f2

        f3 = f2.unsqueeze(1).repeat(1, input.size()[1], 1)

        encoder_input = torch.cat((input, f3), 2).view(batch_size, -1, input.size()[1])
        encoder_input = self.bn(encoder_input).view(batch_size, -1, encoder_input.size()[1])

        encoder_input = self.dropout(encoder_input)
        output, state = self.encoder(encoder_input, hidden[0])

        he = state

        he_reduced = self.fc_h(he.view(batch_size, -1))
        #ce_reduced = self.fc_c(ce.view(batch_size, -1))

        return output, he_reduced, pos_c_state

    def init_hidden(self, batch_size):
        """
        Initial the hidden state
        """
        h0 = Variable(torch.zeros(2, batch_size, self._opts.hidden_size), requires_grad=False)
        c0 = Variable(torch.zeros(2, batch_size, self._opts.hidden_size), requires_grad=False)

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
        self.decoder = nn.GRUCell(self._opts.emb_size, self._opts.hidden_size)

        self._atten_size = 2 * self._opts.hidden_size

        self.input_in = nn.Linear(self._opts.emb_size * 3 + self._opts.hidden_size, self._opts.emb_size)
        #self.input_attn = nn.Linear(self._opts.emb_size, self._opts.hidden_size)

        self.line_in = nn.Linear(self._opts.hidden_size, self._opts.hidden_size * 2)

        self.line_out = nn.Linear(self._atten_size + self._opts.hidden_size, self._opts.vocab_len)
        #self.line_out_2 = nn.Linear( (self._atten_size + self._opts.hidden_size) / 2, self._opts.vocab_len)

        self.logsoftmax = nn.LogSoftmax()


        nn.init.orthogonal(self.decoder.weight_hh)
        nn.init.orthogonal(self.decoder.weight_ih)

    def forward(self, encoder_state, encoder_output, target, input, pos_feature, char_embedding, pre_state=None):
        batch_size = target.size(0)

        hs = Variable(torch.zeros(batch_size, self._opts.hidden_size), requires_grad=False).cuda() if pre_state is None else pre_state
        #hs = encoder_state if pre_state is None else pre_state

        outputs = []
        #argmax = target.squeeze(1)
        # target size is batch_size x seq_len x emb_size
        for i in xrange(target.size()[1]):
            #compute w_s * s_i ------> batch_size * 1 * hidden_size
            decode_input = target[:, i]
            input_tensor = input[:, i] if input.size()[1] > i else Variable(torch.zeros(batch_size, self._opts.emb_size), requires_grad=False).cuda()

            #input_features = self.input_in(torch.cat((decode_input, pos_feature, encoder_state, input_tensor), 1))
            input_features = F.tanh(self.input_in(torch.cat((decode_input, pos_feature, encoder_state, input_tensor), 1)))
            input_features = input_features

            hs = self.decoder(input_features, hs)

            targetT = F.tanh(self.line_in(hs).unsqueeze(2))

            attn = torch.bmm(encoder_output, targetT).squeeze(2)
            softmax_score = F.softmax(F.tanh(attn)).unsqueeze(1)

            h_star = torch.bmm(softmax_score, encoder_output).squeeze(1)

            feature = torch.cat((h_star, hs), 1)

            output = self.logsoftmax((self.line_out(feature)))
            outputs.append(output)

        return hs, outputs
