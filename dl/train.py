from __future__ import division
from collections import OrderedDict
from tqdm import trange

from model import *
from data import Loader
from batcher import Batcher


import optparse
import torch
from torch import optim
import time
import numpy as np 

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
    "-c", "--emb_size", default="15",
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
    "-e", "--eval", default="0",
    type='int', help="train or evaluation"
)
optparser.add_option(
    "-s", "--seed", default="26",
    type='int', help="random seed for CPU and GPU"
)


opts = optparser.parse_args()[0]

train_loader = Loader(opts.train, opts.batch_size)
print train_loader._id_to_char
opts.vocab_len = len(train_loader._char_to_id)
opts.pos_len = len(train_loader._pos_to_id)
opts.max_pos_len = train_loader._pos_max_len
opts.use_cuda = opts.use_cuda == 1
opts.eval = opts.eval == 1
opts.data_size = train_loader.get_data_size()

if not torch.cuda.is_available():
    opts.use_cuda = False

torch.manual_seed(opts.seed)
np.random.seed(opts.seed)

# weights for paddings, set to 0
criterion = nn.NLLLoss()

model = Module(opts)
print model

if opts.use_cuda:
    print "Find GPU enable, using GPU to compute..."
    model.cuda()
    criterion.cuda()
    torch.cuda.manual_seed(opts.seed)
else:
    print "Find GPU unable, using CPU to compute..."

opt = optim.Adadelta(model.parameters(), lr=0.1)
epoch = 10
train_batcher = Batcher(opts.batch_size, train_loader.get_data(), opts.max_pos_len, opts.eval)

# trainning 
for step in xrange(epoch):
    if step != 0:
        torch.save(model, '../model/model%d.pkl'%(step))

    t = trange(int(opts.data_size / opts.batch_size), desc='ML')
    for iter in t:
        input, target, pos, target_length = train_batcher.next()
        input_tensor = Variable(torch.LongTensor(input))
        target_tensor = Variable(torch.LongTensor(target))
        pos_tensor = Variable(torch.LongTensor(pos))

        loss = 0
        accuracy = 0
        predicts = Variable(torch.ones(opts.batch_size, len(target[0])))

        opt.zero_grad()
        if opts.use_cuda:
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()
            pos_tensor = pos_tensor.cuda()
            predicts = predicts.cuda()

        outputs = model(input_tensor, pos_tensor, target_tensor)

        for i in xrange(len(target[0])):
            loss += criterion(outputs[i], target_tensor[:, i])
            _, pred = torch.max(outputs[i], 1)
            predicts[:,i] = pred

        loss.backward()
        opt.step()

        accuracy = torch.sum(torch.eq(predicts.long() ,target_tensor).long())
        t.set_description('ML (loss=%g, accuracy=%g)' % (loss.cpu().data[0] / opts.batch_size, accuracy.cpu().data[0] / (opts.batch_size * len(target[0]))))


