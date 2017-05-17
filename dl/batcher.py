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
        self._data_len = len(self._data)
        if self._eval:
            self._index = np.arange(self._data_len)
        else:
            self._index = np.random.permutation(self._data_len)
        self._max_pos_len = max_pos_len
        self._curser = 0
    
    def next(self):
        """
        return next batch data
        """
        input = []
        pos = []
        target = []
        target_length = []
        input_length = []
        data_moved = self._batch_size
        if self._curser == self._data_len:
            if not self._eval:
                self._index = np.random.permutation(self._data_len)
            self._curser = 0
        elif self._curser + self._batch_size > self._data_len:
            diff = self._data_len - self._curser
            for i in self._index[self._curser : self._curser + diff]:
                input_length.append(self._data[i]['input'])
                target_length.append(len(self._data[i]['label']))
                input.append(self._data[i]['input'])
                pos.append(self._data[i]['pos'])
                target.append(self._data[i]['label'])
            self._curser = 0
            if not self._eval:
                self._index = np.random.permutation(self._data_len)
                for i in self._index[self._curser : self._curser + self._batch_size - diff]:
                    input_length.append(self._data[i]['input'])
                    target_length.append(len(self._data[i]['label']))
                    input.append(self._data[i]['input'])
                    pos.append(self._data[i]['pos'])
                    target.append(self._data[i]['label'])
                data_moved = self._batch_size - diff

            input = self.padding(input)
            target = self.padding(target)
            pos = self.padding(pos, self._max_pos_len)
            self._curser += data_moved

            return input, target, pos, target_length, input_length 

        for i in self._index[self._curser : self._curser + self._batch_size]:
            input_length.append(self._data[i]['input'])
            target_length.append(len(self._data[i]['label']))
            input.append(self._data[i]['input'])
            pos.append(self._data[i]['pos'])
            target.append(self._data[i]['label'])

        input = self.padding(input)
        target = self.padding(target)
        pos = self.padding(pos, self._max_pos_len)
        self._curser += data_moved

        return input, target, pos, target_length, input_length

    def padding(self, input, maxlength=0):
        """
        padding the sequence to the longest seq
        """
        maxlen = maxlength if maxlength != 0 else len(max(input, key=len))
        for i in xrange(self._batch_size):
            input[i] += [0] * (maxlen - len(input[i]))
        return input
