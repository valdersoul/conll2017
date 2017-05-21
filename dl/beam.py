from copy import deepcopy
import numpy as np

import torch
from torch.autograd import Variable



class Hyper(object):
    """
    Class to store the decoded sentence and its possibility
    """
    def __init__(self, word, log_prob, state, hyper=None):
        self.word_list = []
        self.log_prob = 0
        self.state = state
        if hyper == None:
            self.word_list = []
            self.word_list.append(word)
            self.log_prob = log_prob
        else:
            self.word_list = deepcopy(hyper.word_list)
            self.word_list.append(word)
            self.log_prob = hyper.log_prob + log_prob

    def insert(self, word, log_prob):
        self.word_list.append(word)
        self.log_prob += log_prob

    def lastword(self):
        return self.word_list[-1]

    def remove_last(self):
        self.word_list.pop()

class Beam(object):
    """
    Beam Searcher
    """
    def __init__(self, model, beam_size, max_step, encoder_state, encoder_output, pos_feature, start_decode, input):
        self.max_step = max_step
        self.beam_size = beam_size
        self.model = model
        self.encoder_state = encoder_state
        self.encoder_output = encoder_output
        self.pos_feature = pos_feature
        self.source = deepcopy(input[0])
        self.hyper_lise = []

        prob, index, state = self._get_list(start_decode, Variable(torch.LongTensor([self._next_input()])).cuda().unsqueeze(1))
        self.hyper_lise = self._generate_list(index, prob, state)

    def _next_input(self):
        if not self.source:
            return 0
        else:
            return self.source.pop(0) 

    def run(self):
        results = []
        step = 0
        while len(results) < self.beam_size and step < self.max_step:
            temp_hyper = []
            next_input = Variable(torch.LongTensor([self._next_input()])).cuda().unsqueeze(1)
            for hyper in self.hyper_lise:
                prob, index, pre_state = self._get_list(Variable(torch.LongTensor([hyper.lastword()])).cuda().unsqueeze(1), next_input, pre_state=hyper.state)
                temp_hyper += self._generate_list(index, prob, pre_state, hyper=hyper)

            self.hyper_lise = temp_hyper
            self._sorthyper()

            temp_next = []
            for hyper in self.hyper_lise:
                if hyper.lastword() == 3:
                    results.append(hyper)
                else:
                    temp_next.append(hyper)

            self.hyper_lise = temp_next
            step += 1

        self.hyper_lise = results if len(results) != 0 else self.hyper_lise
        self._sorthyper()
        final = self.hyper_lise[-1]
        final.remove_last()
        return final

    def _sorthyper(self):
        sorted_hyper = sorted(self.hyper_lise, key=lambda hyper: hyper.log_prob/len(hyper.word_list))
        self.hyper_lise = sorted_hyper[-self.beam_size:]

    def _get_list(self, word, input, topk=True, pre_state=None):
        state, output = self.model.decode_once(self.encoder_state, self.encoder_output, word, input, self.pos_feature, pre_state = pre_state)
        if not topk:
            prob, index = torch.topk(output[0], output[0].size()[1])
        else:
            prob, index = torch.topk(output[0], self.beam_size)
        prob = prob.data[0].tolist()
        index = index.data[0].tolist()
        return prob, index, state

    def _generate_list(self, index, prob, pre_state, hyper=None):
        temp = []
        for log_prob, word in zip(prob, index):
            temp.append(Hyper(word, log_prob, pre_state, hyper))
        return temp
