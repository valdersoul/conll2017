from __future__ import division
from collections import OrderedDict

from model import *
from data import Loader
from batcher import Batcher
import optparse
import torch

# Read parameters from command line
optparser = optparse.OptionParser()

optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-c", "--emb_size", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--hidden_size", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--batch_size", default="16",
    type='int', help="batch_size"
)
optparser.add_option(
    "-g", "--use_cuda", default="1",
    type='int', help="whether use gpu"
)
optparser.add_option(
    "-m", "--run_type", default="0",
    type='int', help="train or evaluation"
)



opts = optparser.parse_args()[0]

train_loader = Loader(opts.train, opts.batch_size)
opts.vocab_len = len(train_loader._char_to_id)
opts.pos_len = len(train_loader._pos_to_id)
opts.max_pos_len = train_loader._pos_max_len
opts.use_cuda = opts.use_cuda == 1
opts.run_type = opts.run_type == 1

if not torch.cuda.is_available():
    opts.use_cuda = False

train_batcher = Batcher(opts.batch_size, train_loader._get_data(), opts.max_pos_len)


input, target, pos = train_batcher.next()

model = Module(opts)
print model
if opts.use_cuda:
    model.cuda()
    output, hidden = model(input.cuda(), pos.cuda(), target.cuda())
else:
    output, hidden = model(input, pos, target)

print hidden[0].size()
print output.size()



