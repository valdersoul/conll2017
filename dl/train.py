from __future__ import division
from collections import OrderedDict

from model import *
from data import Loader
from batcher import Batcher
import optparse
import torch
from torch import optim
import time

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

# weights for paddings, set to 0
loss_weights = torch.ones(opts.vocab_len)
loss_weights[0] = 0
criterion = nn.NLLLoss(loss_weights)

model = Module(opts)
print model

if opts.use_cuda:
    print "Find GPU enable, using GPU to compute..."
    model.cuda()
    criterion.cuda()
else:
    print "Find GPU unable, using CPU to compute..."

opt = optim.Adadelta(model.parameters(), lr = 0.1)

for step in xrange(int(100 * 10000 / 16)):
    if step % (10000 / 16) == 0 and step != 0:
        torch.save(model, '../model/model%d.pkl' %(step) )
    train_batcher = Batcher(opts.batch_size, train_loader._get_data(), opts.max_pos_len)
    input, target, pos = train_batcher.next()
    input_tensor = Variable(torch.LongTensor(input))
    target_tensor = Variable(torch.LongTensor(target))
    pos_tensor = Variable(torch.LongTensor(pos))

    opt.zero_grad()

    if opts.use_cuda:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        pos_tensor = pos_tensor.cuda()

    loss = 0
    outputs = model(input_tensor, pos_tensor, target_tensor)
    for i in xrange(len(target[0])):
        loss += criterion(outputs[i], target_tensor[:,i])

    loss.backward()
    opt.step()
    if step % 100 == 0 and step != 0:
        print loss.cpu().data[0] / opts.batch_size


