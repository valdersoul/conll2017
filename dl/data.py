import numpy
import codecs

WORD_START = '<w>'
WORD_END = '</w>'

UNKNOWN_CHARACTER = '[UNK]'
PADDING = '[PAD]'
START_DECODING = '[START]'
END_DECODING = '[END]'
MAX_COUNT = 10000000

class Loader(object):
    '''A data loader for tringing and testing'''
    def __init__(self, file_path, c2i=[], i2c=[], p2i=[], i2p=[]):
        self._file_path = file_path

        lines = [line.strip() for line in codecs.open(self._file_path, 'r', encoding = 'utf-8')]
        self._lemma = []
        self._form = []
        self._pos = []

        self._char_max_len = 0
        self._pos_max_len = 0

        for l in lines:
            t1, t2, t3 = l.split(u'\t')
            self._lemma += [w for w in t1]
            self._form += [w for w in t2]
            self._pos += t3.split(';')
            if len(t2) > self._char_max_len:
                self._char_max_len = len(t2)
            if len(t3.split(';')) > self._pos_max_len:
                self._pos_max_len = len(t3.split(';'))

        self._raw_data = lines

        if not c2i:
            vocab = self._lemma + self._form
            char_dico = self._create_dico(vocab)
            pos_dico = self._create_dico(self._pos)

            char_dico[UNKNOWN_CHARACTER] = MAX_COUNT - 1
            char_dico[PADDING] = MAX_COUNT
            char_dico[START_DECODING] = MAX_COUNT - 2
            char_dico[END_DECODING] = MAX_COUNT - 3

            pos_dico[UNKNOWN_CHARACTER] = MAX_COUNT - 1
            pos_dico[PADDING] = MAX_COUNT

            self._char_to_id, self._id_to_char = self._create_mapping(char_dico)
            self._pos_to_id, self._id_to_pos = self._create_mapping(pos_dico)
        else:
            self._char_to_id = c2i
            self._id_to_char = i2c
            self._pos_to_id = p2i
            self._id_to_pos = i2p

        self._data = self._prepare_data()


    def get_data(self):
        return self._data

    def get_mappings(self):
        return self._char_to_id, self._id_to_char, self._pos_to_id, self._id_to_pos

    def get_data_size(self):
        return len(self._data)

    def _prepare_data(self):
        data = []
        for line in self._raw_data:
            l = line.split('\t')
            input_seq = [self._char_to_id[c if c in self._char_to_id else UNKNOWN_CHARACTER] for c in l[0]]
            label_seq = [self._char_to_id[START_DECODING]] + [self._char_to_id[c if c in self._char_to_id else UNKNOWN_CHARACTER] for c in l[1]] + [self._char_to_id[END_DECODING]]
            pos = [self._pos_to_id[p if p in self._pos_to_id else UNKNOWN_CHARACTER] for p in l[2].split(';')[1:]]
            data.append(
                {
                    'input' : input_seq,
                    'label' : label_seq,
                    'pos'   : pos
                }
            )
        return data

    def _create_dico(self, item_list):
        """
        Create a dictionary of items from a list of list of items.
        """
        assert type(item_list) is list
        dico = {}
        for item in item_list:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
        return dico

    def _create_mapping(self, dico):
        """
        Create a mapping (item to ID / ID to item) from a dictionary.
        Items are ordered by decreasing frequency.
        """
        sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
        id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
        item_to_id = {v: k for k, v in id_to_item.items()}
        return item_to_id, id_to_item

def main():
    """
    test main function
    """
    test = Loader('../all/task1/albanian-train-high', 16, None)


if __name__ == '__main__':
    main()
