import numpy

WORD_START = '<w>'
WORD_END = '</w>'

UNKNOWN_CHARACTER = '[UNK]'
PADDING = '[PAD]'
START_DECODING = '[START]'
END_DECODING = '[END]'

class Loader(object):
    '''A data loader for tringing and testing'''
    def __init__(self, file_path, batch_size, opts):
        self._file_path = file_path
        self._batch_size = batch_size
        self._opts = opts

        