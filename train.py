'''
This script handling the training process.
'''

import argparse
import math
import time

from subprocess import Popen
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from DataLoader import DataLoader

def get_performance(crit, logsoftmax, pred, gold, opt, smoothing_eps=0.1, num_class=None):
    ''' Apply label smoothing if needed '''

    # TODO: Add smoothing
    # if smoothing:
    #     assert bool(num_class)
    #     eps = 0.1
    #     gold = gold * (1 - eps) + (1 - gold) * eps / num_class
    #     raise NotImplementedError

    gold = gold.contiguous().view(-1)

    #logsoftmax = nn.LogSoftmax()
    #pred = logsoftmax(pred)

    loss = crit(pred, gold)

    if smoothing_eps:
        if opt.cuda:
            smooth = gold.ne(Constants.PAD).type(torch.cuda.FloatTensor)*torch.mean(pred, -1)
        else:
            smooth = gold.ne(Constants.PAD).type(torch.FloatTensor)*torch.mean(pred, -1)
        smooth = -smooth.sum()
        loss = (1-smoothing_eps)*loss + smoothing_eps*smooth
        

    

   

    pred = pred.max(1)[1]

    
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum()

    return loss, n_correct

def train_epoch(model, training_data, crit, logsoftmax, optimizer, opt):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src, tgt = batch
        gold = tgt[0][:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src, tgt)

        # backward
        loss, n_correct = get_performance(crit, logsoftmax, pred, gold, opt)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0]

    return total_loss/n_total_words, n_total_correct/n_total_words

def eval_epoch(model, validation_data, crit, logsoftmax, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(
            validation_data, mininterval=2,
            desc='  - (Validation) ', leave=False):

        # prepare data
        src, tgt = batch
        gold = tgt[0][:, 1:]

        # forward
        pred = model(src, tgt)
        loss, n_correct = get_performance(crit, logsoftmax, pred, gold, opt)

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0]

    return total_loss/n_total_words, n_total_correct/n_total_words

def train(model, training_data, validation_data, crit, logsoftmax, optimizer, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    p_validation = None
    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, crit, logsoftmax, optimizer, opt)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, crit, logsoftmax, opt)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

            ###########################################################################################
            if opt.external_validation_script:
                print("Calling external validation script")
                if p_validation is not None and p_validation.poll() is None:
                    print("Waiting for previous validation run to finish")
                    print("If this takes too long, consider increasing validation interval, reducing validation set size, or speeding up validation by using multiple processes")
                    valid_wait_start = time.time()
                    p_validation.wait()
                    print("Waited for {0:.1f} seconds".format(time.time()-valid_wait_start))
                external_validation_script = [opt.external_validation_script[0], model_name, opt.external_validation_script[1], opt.external_validation_script[2], opt.data]
                p_validation = Popen(external_validation_script)


            ###########################################################################################

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='all')

    parser.add_argument('-no_cuda', action='store_true')

    parser.add_argument('-multi_gpu', action='store_true')

    parser.add_argument('-external_validation_script', type=str, default=None, metavar='PATH', nargs='*',
                         help="location of validation script (to run your favorite metric for validation) (default: %(default)s)")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    #========= Preparing DataLoader =========#
    training_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_insts=data['train']['src'],
        tgt_insts=data['train']['tgt'],
        batch_size=opt.batch_size,
        cuda=opt.cuda)

    validation_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_insts=data['valid']['src'],
        tgt_insts=data['valid']['tgt'],
        batch_size=opt.batch_size,
        shuffle=False,
        cuda=opt.cuda)

    opt.src_vocab_size = training_data.src_vocab_size
    opt.tgt_vocab_size = training_data.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight and training_data.src_word2idx != training_data.tgt_word2idx:
        print('[Warning]',
              'The src/tgt word2idx table are different but asked to share word embedding.')

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        proj_share_weight=opt.proj_share_weight,
        embs_share_weight=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)


    if opt.multi_gpu:
        transformer = nn.DataParallel(transformer)

    #print(transformer)

    # optimizer = ScheduledOptim(
    #     optim.Adam(
    #         transformer.get_trainable_parameters(),
    #         betas=(0.9, 0.98), eps=1e-09),
    #     opt.d_model, opt.n_warmup_steps)

    optimizer = ScheduledOptim(
        optim.Adam(
            transformer.parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)


    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        #return nn.CrossEntropyLoss(weight, size_average=False)
        return nn.NLLLoss(weight, size_average=False)

    crit = get_criterion(training_data.tgt_vocab_size)

    logsoftmax = nn.LogSoftmax()

    if opt.cuda:
        transformer = transformer.cuda()
        crit = crit.cuda()
        logsoftmax = logsoftmax.cuda()

    train(transformer, training_data, validation_data, crit, logsoftmax, optimizer, opt)

if __name__ == '__main__':
    main()
