from data import Loader
import numpy as np

import torch

import torch.nn.utils as utils
from torch.autograd import Variable

class Batcher(object):
    """
    batch loader for the data
    """
    def __init__(self, batch_size, data, max_pos_len, eval):
        self._batch_size = batch_size
        self._data = data
        self._eval = eval
        if self._eval:
            self._index = np.arange(len(self._data))
        else:
            self._index = np.random.permutation(len(self._data))
        self._max_pos_len = max_pos_len
        self._curser = 0

    def next(self):
        """
        return next batch data
        """
        input = []
        pos = []
        target = []

        if self._curser == len(self._data):
            if not self._eval:
                self._index = np.random.permutation(len(self._data))
            self._curser = 0

        for i in self._index[self._curser : self._curser + self._batch_size]:
            input.append(self._data[i]['input'])
            pos.append(self._data[i]['pos'])
            target.append(self._data[i]['label'])

        input = self.padding(input)
        target = self.padding(target)
        pos = self.padding(pos, self._max_pos_len)
        self._curser += self._batch_size
        
        return input, target, pos
        
    def padding(self, input, maxlength=0):
        """
        padding the sequence to the longest seq
        """
        maxlen = maxlength if maxlength != 0 else len(max(input, key=len))
        for i in xrange(self._batch_size):
            input[i] += [0] * (maxlen - len(input[i]))
        return input
