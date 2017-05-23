from __future__ import division

import math
import optparse
import time
import numpy as np
import codecs
import os

from collections import OrderedDict
from tqdm import trange
from model import Module
from data import Loader
from batcher import Batcher
from beam import Beam, Hyper
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable

# Read parameters from command line
optparser = optparse.OptionParser()

optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-D", "--dev", default="",
    help="develop set location"
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
optparser.add_option(
    "-d", "--dropout", default="0.5",
    type='float', help="dropout ratio"
)
optparser.add_option(
    "-B", "--beam_size", default="4",
    type='int', help="beam search size"
)
optparser.add_option(
    "-r", "--model_path", default="",
    help="reload model location"
)
optparser.add_option(
    "-R", "--result_file", default="",
    help="result file location"
)
optparser.add_option(
     "--clip", default="5",
    type='float', help="result file location"
)

decode_batch = 2

def start_train(model, criterion, opts, train_batcher, dev_batcher):
    """
    Training the model
    """
    if opts.use_cuda:
        print "Find GPU enable, using GPU to compute..."
        model.cuda()
        criterion.cuda()
        torch.cuda.manual_seed(opts.seed)
    else:
        print "Find GPU unable, using CPU to compute..."

    opt = optim.Adam(model.parameters(), weight_decay=0.0001)
    epoch = 500
    devLoss = 100
    best_step = 0
    # trainning
    for step in xrange(epoch):
        if step % 50 == 0 and step != 0:
            torch.save(model, "../model/model%d.pkl"%(step))
        total_loss = []
        t = trange(int(math.ceil(opts.data_size / opts.batch_size)), desc='ML')
        for iter in t:
            input, target, pos, target_length, input_length = train_batcher.next()
            input_tensor = Variable(torch.LongTensor(input))
            target_tensor = Variable(torch.LongTensor(target))
            pos_tensor = Variable(torch.LongTensor(pos))

            loss_t = 0
            accuracy = 0
            loss = []
            opt.zero_grad()
            if opts.use_cuda:
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()
                pos_tensor = pos_tensor.cuda()

            _, outputs = model(input_tensor, pos_tensor, target_tensor)

            for i in xrange(len(target[0]) - 1):
                label = target_tensor[:, i + 1].contiguous().view(-1)
                loss_t += criterion(outputs[i], label)

            loss_t /= (np.sum(target_length) - opts.batch_size)
            total_loss.append(loss_t.cpu().data[0])
            loss_t.backward()
            nn.utils.clip_grad_norm(model.parameters(), opts.clip)
            opt.step()

            t.set_description('Iter%d (loss=%g)' % (step, loss_t.cpu().data[0]))
        print 'Average loss: %g' % (np.mean(total_loss))
        ave_dev_loss = eval(model, dev_batcher, criterion)
        if ave_dev_loss < devLoss:
            print 'New best dev loss: %g' % (ave_dev_loss)
            print 'Saving model ...'
            torch.save(model, '../model/model%s.pkl'%(opts.train.split('/')[-1]))
            devLoss = ave_dev_loss
            best_step = step
        if step - best_step > 30:
            print 'Eearly stopping, no update after 30 epoches'
            break
        model.train()

def eval(model, dev_batcher, criterion):
    model.eval()
    t = trange(int(dev_batcher._data_len / decode_batch), desc='ML')
    total_correct = []
    for iter in t:
        input, target, pos, target_length, input_length = dev_batcher.next()
        input_tensor = Variable(torch.LongTensor(input))
        target_tensor = Variable(torch.LongTensor(target))
        pos_tensor = Variable(torch.LongTensor(pos))

        accuracy = 0
        loss_t = 0

        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        pos_tensor = pos_tensor.cuda()

        _, outputs = model(input_tensor, pos_tensor, target_tensor)

        for i in xrange(len(target[0]) - 1):
            label = target_tensor[:, i + 1].contiguous().view(-1)
            loss_t += criterion(outputs[i], label)
        loss_t /= (np.sum(target_length) - decode_batch)
        loss_t = loss_t.cpu().data[0]
        t.set_description('Evaluating (loss=%g)' % (loss_t))

        total_correct.append(loss_t)

    ave_loss = np.mean(total_correct)
    print 'Average loss: %g' % (ave_loss)
    return ave_loss

def get_accuracy(predicts, target_tensor):
    unmask = target_tensor[:, 1:] == 0
    mask = target_tensor[:, 1:] != 0

    total_num = (torch.sum(mask.long())).cpu().data[0]
    zero_num = (torch.sum(unmask.long())).cpu().data[0]

    predicts_mask = torch.zeros(target_tensor[:, 1:].size()).cuda()
    predicts_mask.masked_copy_(mask.data, predicts.data).long()
    predicts_mask = Variable(predicts_mask.long())

    correct_mask = predicts_mask == target_tensor[:, 1:].long()
    correct_num = torch.sum(correct_mask).cpu().data[0]

    correct = (correct_num - zero_num) / total_num
    return correct


def decode(model, opts, test_batcher, i2c, i2p):
    """
    Decode the input
    """
    if opts.use_cuda:
        print "Find GPU enable, using GPU to compute..."
        model.cuda()
        torch.cuda.manual_seed(opts.seed)
    else:
        print "Find GPU unable, using CPU to compute..."

    result_file = opts.result_file
    result_writer = codecs.open(result_file, 'w', 'utf-8')

    t = trange(opts.data_size, desc='DECODE')
    for iter in t:
        input, target, pos, target_length, input_length = test_batcher.next()
        input_tensor = Variable(torch.LongTensor(input))
        target_tensor = Variable(torch.LongTensor(target))
        pos_tensor = Variable(torch.LongTensor(pos))
        if opts.use_cuda:
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()
            pos_tensor = pos_tensor.cuda()

        encoder_state, encoder_output, pos_feature,_ = model.encode_once(input_tensor, pos_tensor)
        start_decode = Variable(torch.LongTensor([2])).cuda().unsqueeze(1)

        beam = Beam(model, opts.beam_size, opts.max_target_len, encoder_state, encoder_output, pos_feature, start_decode, input)
        hyper = beam.run()

        raw_input = ""
        result = ""
        raw_pos = ""
        for word in hyper.word_list:
            result += i2c[word]
        for word in input[0]:
            raw_input += i2c[word]
        for word in pos[0]:
            if word == 0:
                break
            raw_pos += i2p[word]+";"
        raw_pos = raw_pos[:-1]
        result_writer.write(raw_input + '\t' + result + '\t' + raw_pos + '\n')

    result_writer.close()
    os.system(('../evaluation/evalm.py --gold %s --guess %s --task 1')%(opts.test, opts.result_file))

def main(): 
    opts = optparser.parse_args()[0]

    train_loader = Loader(opts.train)

    opts.vocab_len = len(train_loader._char_to_id)
    opts.pos_len = len(train_loader._pos_to_id)
    opts.max_pos_len = train_loader._pos_max_len
    opts.max_target_len = train_loader._char_max_len
    opts.use_cuda = opts.use_cuda == 1
    opts.eval = opts.eval == 1
    opts.data_size = train_loader.get_data_size()

    if not torch.cuda.is_available():
        opts.use_cuda = False
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    if not opts.eval:
        # weights for paddings, set to 0
        loss_weights = torch.ones(opts.vocab_len)
        loss_weights[0] = 0
        criterion = nn.NLLLoss(loss_weights, size_average=False)
        model = Module(opts)
        if opts.model_path is not '':
            model = torch.load(opts.model_path)
        train_batcher = Batcher(opts.batch_size, train_loader.get_data(), opts.max_pos_len, opts.eval)

        c2i, i2c, p2i, i2p = train_loader.get_mappings()
        dev_loader = Loader(opts.dev, c2i, i2c, p2i, i2p)
        dev_batcher = Batcher(decode_batch, dev_loader.get_data(), opts.max_pos_len, True)

        print model
        start_train(model, criterion, opts, train_batcher, dev_batcher)
    else:
        model = torch.load(opts.model_path)
        model.eval()
        print model

        c2i, i2c, p2i, i2p = train_loader.get_mappings()

        test_loader = Loader(opts.test, c2i, i2c, p2i, i2p)
        test_batcher = Batcher(1, test_loader.get_data(), opts.max_pos_len, opts.eval)

        opts.data_size = test_loader.get_data_size()
        print i2c
        decode(model, opts, test_batcher, i2c, i2p)

if __name__ == '__main__':
    main()
