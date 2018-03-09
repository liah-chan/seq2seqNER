# import mkl
# mkl.set_num_threads(16)
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import math
import time
import os
import argparse
from argparse import RawTextHelpFormatter
import configparser
import sys
from pprint import pprint
import numpy as np
import subprocess
from data_loader import *
from model import *
from evaluate import *
from plot_utils import *
import tensor_utils
from collections import defaultdict, OrderedDict
from itertools import chain
# this is for testing only BiLSTM
#input_lang in parameter is actually not needed

use_cuda = torch.cuda.is_available()

def train_step(parameters, input_lang, input_variable, target_variable, lengths_sorted,
               encoder, optimizer, criterion, eval_during_train =True,
               input_variable_char=None):
    optimizer.zero_grad()
    # decoder_optimizer.zero_grad()
    loss = 0
    # the input_variable is of shape(seq_len x batch_size)
    input_batch_size, input_seq_len = input_variable.size()[1], input_variable.size()[0]
    target_batch_size, target_seq_len = target_variable.size()[1], target_variable.size()[0]
    assert input_batch_size == target_batch_size
    assert input_seq_len == target_seq_len

    encoder.train()
    encoder_hidden = encoder.init_hidden(input_batch_size)

    encoder_outputs, encoder_hidden, enc_output_lengths = encoder(input_variable,
                                                                  input_lengths=lengths_sorted,
                                                                  hidden = encoder_hidden,
                                                                  input_seqs_char = input_variable_char)

    all_decoder_outputs = Variable(torch.zeros(target_seq_len, input_batch_size, encoder.output_size))

    if use_cuda:
        all_decoder_outputs = all_decoder_outputs.cuda()
    
    if eval_during_train :
        decoded_word_idxes = np.zeros((input_seq_len, input_batch_size), dtype=int)
        
    for i in range(encoder_outputs.size(0)): #loop over seq_len
        all_decoder_outputs[i] = encoder_outputs[i, :, :]
        # try:
        loss += criterion(encoder_outputs[i, :, :], target_variable[i, :])
        if eval_during_train:
            _, topi = encoder_outputs.data.topk(1, dim=-1)
            decoded_word = topi[i].cpu().numpy()
            decoded_word_idxes[i,:] = decoded_word.squeeze(-1)

    loss.backward()

    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), parameters['clip'])

    optimizer.step()
    decoded_word_idxes = np.transpose(decoded_word_idxes, (1, 0))
    return decoded_word_idxes, loss.data[0], ec, None


def train_step_seq2seq(parameters, output_lang, input_variable, target_variable,
                       lengths_sorted, encoder, decoder, optimizer,
                       criterion, eval_during_train=True, input_variable_char=None):
    optimizer.zero_grad()
    loss = 0
    input_batch_size, input_seq_len = input_variable.size()[1], input_variable.size()[0]
    target_batch_size, target_seq_len = target_variable.size()[1], target_variable.size()[0]
    assert input_batch_size == target_batch_size
    assert input_seq_len == target_seq_len
    
    encoder.train()
    decoder.train()
    encoder_hidden = encoder.init_hidden(input_batch_size)
    encoder_outputs, encoder_hidden, enc_output_lengths = encoder(input_variable,
                                                                  input_lengths=lengths_sorted,
                                                                  hidden=encoder_hidden,
                                                                  input_seqs_char=input_variable_char)
    
    decoder_input = Variable(torch.LongTensor([output_lang.word2idx['<SOS>']] * input_batch_size).unsqueeze(0))
    last_hidden = (encoder_hidden[0][1].unsqueeze(0), encoder_hidden[1][1].unsqueeze(0))
    all_decoder_outputs = Variable(torch.zeros(target_seq_len, input_batch_size, decoder.output_size))
    
    if use_cuda:
        all_decoder_outputs = all_decoder_outputs.cuda()
        decoder_input = decoder_input.cuda()

    decoder_attentions = torch.zeros(target_seq_len + 1, input_seq_len + 1)
    
    if eval_during_train:
        decoded_word_idxes = np.zeros((input_seq_len, input_batch_size), dtype=int)
    
    use_teacher_forcing = True if random.random() < parameters['teacher_forcing_ratio'] else False
    if use_teacher_forcing:
        for di in range(target_seq_len):
            decoder_output, last_hidden, decoder_attention = decoder(di,
                                                                        decoder_input, last_hidden, encoder_outputs,
                                                                        input_lengths=lengths_sorted)
            # TODO: all_decoder_outputs doesn't seem to be used
            all_decoder_outputs[di] = decoder_output
            decoder_input = target_variable[di].unsqueeze(0)  # Teacher forcing
            # loss += criterion(decoder_output, target_variable[di, :])
            # for i in range(decoder_output.size(1)):
            loss += criterion(decoder_output[0], target_variable[di, :])
            
            if eval_during_train:
                _, topi = decoder_output.data.topk(1, dim=-1)
                decoded_word = topi.cpu().numpy()
                decoded_word_idxes[di] = decoded_word.squeeze(-1).squeeze(0)
              
    loss.backward()
    
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), parameters['clip'])
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), parameters['clip'])
    
    # encoder_optimizer.step()
    optimizer.step()

    # if not eval_during_train, it would be empty
    decoded_word_idxes = np.transpose(decoded_word_idxes, (1, 0))
    return decoded_word_idxes, loss.data[0], ec, dc
    # return loss.data[0], ec, dc
    
def model_selection(parameters, input_lang, output_lang ):
    # encoder, decoder, encoder_optimizer, decoder_optimizer = None, None, None, None
    encoder, decoder, optimizer =  None, None, None
    if parameters['step'] == 1 or (parameters['step'] == 2 and parameters['resume'] != 1):
        # step 1 or step 2 baseline
        print('Training in step {:d}, initializing model at random...'.format(parameters['step']))
        if parameters['use_pretrained_word_embedding']:
            pretraind_embed, emb_size = load_pretrained_word_embeddings(parameters, input_lang)
        else:
            pretraind_embed = None
            emb_size = parameters['hidden_size']
        
        if parameters['nn_model'] == 'seq2seq':
            encoder = EncoderRNN(input_lang.n_words + 1, emb_size,
                                 parameters['hidden_size'], parameters['n_layers'],
                                 dropout=parameters['dropout_p'], pretrained_weights=pretraind_embed,
                                 use_char_embedding=parameters['use_char_embedding'],
                                 char_alphabet_size=input_lang.n_chars + 1,
                                 char_emb_size=parameters['char_emb_size']
                                 )
            decoder = BahdanauAttnDecoderRNN(output_size=output_lang.n_words + 1,
                                             hidden_size=parameters['hidden_size'],
                                             n_layers=parameters['n_layers'],
                                             # attn_mode=parameters['attn_mode'],
                                             dropout_p=parameters['dropout_p'])
  
            all_params = chain( encoder.parameters() , decoder.parameters())

        elif parameters['nn_model'] == 'bilstm':
            encoder = BiLSTM(input_lang.n_words + 1, emb_size,
                             parameters['hidden_size'], parameters['n_layers'],
                             dropout=parameters['dropout_p'],
                             pretrained_weights=pretraind_embed,
                             output_size=len(output_lang.words) + 1,
                             use_char_embedding=parameters['use_char_embedding'],
                             char_alphabet_size=input_lang.n_chars + 1,
                             char_emb_size=parameters['char_emb_size']
                             )
            all_params = encoder.parameters()
            
    elif parameters['step'] == 2 and parameters['add_class'] == 1:
        print('Training in step {:d}, loading pretrained model...'.format(parameters['step']))
        print('load pretrained encoder at {:s}'.format(parameters['encoder_path']))
        encoder = torch.load(parameters['encoder_path'])
        print(encoder)
        if parameters['nn_model'] == 'seq2seq':
            print('load pretrained decoder at {:s}'.format(parameters['decoder_path']))
            decoder = torch.load(parameters['decoder_path'])
            print(decoder)
            last_w = decoder.out.weight.data.cpu().numpy()
            print(last_w.shape)
            print(last_w)
        else:
            decoder = None
        
        if parameters['add_class'] and parameters['nn_model'] == 'seq2seq':
            decoder.output_size = len(output_lang.words) + 1
            decoder.embedding = nn.Embedding(decoder.output_size, parameters['hidden_size'])
            decoder.out = nn.Linear(2 * parameters['hidden_size'], decoder.output_size)
        
        elif parameters['add_class'] and parameters['nn_model'] == 'bilstm':
            encoder.output_size = len(output_lang.words) + 1
            encoder.out = nn.Linear(2 * parameters['hidden_size'], encoder.output_size)
        
        all_params = encoder.parameters()
        if parameters['nn_model'] == 'seq2seq':
            all_params = chain( encoder.parameters() , decoder.parameters())

    elif parameters['step'] in [1, 2] and parameters['resume'] == 1 and parameters['add_class'] != 1:
        print('Training in step {:d}, resume training, loading pretrained model...'.format(parameters['step']))
        print('load pretrained encoder at {:s}'.format(parameters['encoder_path']))
        encoder = torch.load(parameters['encoder_path'])
        print(encoder)
        all_params = encoder.parameters()
        
        if parameters['nn_model'] == 'seq2seq':
            print('load pretrained decoder at {:s}'.format(parameters['decoder_path']))
            decoder = torch.load(parameters['decoder_path'])
            print(decoder)
            all_params = chain( encoder.parameters() , decoder.parameters())

        else:
            decoder = None
    
    if use_cuda:
        encoder.cuda()
        if decoder is not None:
            decoder.cuda()
    print(encoder)
    if decoder is not None:
        print(decoder)

    optimizer = optim.SGD(all_params, lr=parameters['learning_rate'], momentum=0.8)
    return encoder, decoder, optimizer

def train(parameters, input_lang, output_lang, all_pairs, all_max_len, all_lengths, delete_after_train = False):

    encoder, decoder, optimizer = model_selection(parameters, input_lang, output_lang)
    criterion = nn.NLLLoss(ignore_index=0)
    
    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    ecs, dcs = [], []
    eca, dca = 0, 0
    print_loss_total, plot_loss_total = 0, 0
    all_decoded_word_idxes = np.zeros((len(all_pairs['train']), all_max_len['train']))
    
    # f1s = defaultdict(dict)
    f1s_conll = defaultdict(dict)
    f1s_sklearn = defaultdict(dict)
    for dataset_type in ['train', 'valid', 'test']:
        f1s_conll[dataset_type]['overall'] = []
        f1s_sklearn[dataset_type]['overall'] = []
        
    best_valid_f1 = 0.0
    patience_cnt = 0
    try:
        for epoch in range(1, parameters['n_epochs'] + 1):
            print('\nStarting epoch {:d}...'.format(epoch))
            # unk_idx = input_lang.word2idx['<UNK>']
            epoch_start_time = time.time()
            for i_batch, ((input_variables, target_variables, input_variables_char), batch_lengths) in \
                    enumerate(get_batch(input_lang, output_lang, all_pairs['train'], \
                                        batch_size=parameters['batch_size'],
                                        max_len=all_max_len['train'], lengths=all_lengths['train'])):
                input_variables_sorted, input_variables_char_sorted, target_variables_sorted, \
                lengths_sorted, lengths_argsort \
                    = tensor_utils.sort_variables_lengths(input_variables, target_variables,
                                                          batch_lengths, needs_argsort = True,
                                                          input_variables_char = input_variables_char)
                
                if parameters['nn_model'] == 'seq2seq':
                    decoded_word_idxes, loss, ec, dc = train_step_seq2seq(parameters, output_lang, input_variables_sorted,
                                                              target_variables_sorted, lengths_sorted,
                                                              encoder, decoder, optimizer,criterion,
                                                              input_variable_char = input_variables_char_sorted)
                elif parameters['nn_model'] == 'bilstm':
                    decoded_word_idxes, loss, ec, dc = train_step(parameters, output_lang, input_variables_sorted,
                                                              target_variables_sorted, lengths_sorted,
                                                              encoder, optimizer, criterion,
                                                              input_variable_char = input_variables_char_sorted)

                decoded_word_idxes_new = tensor_utils.sort_results_back(decoded_word_idxes, lengths_argsort)
                all_decoded_word_idxes[i_batch * parameters['batch_size']: \
                    min((i_batch + 1) * parameters['batch_size'], len(all_pairs['train'])), :] = decoded_word_idxes_new
                # Keep track of loss
                print_loss_total += loss
                plot_loss_total += loss
                eca += ec
                if dc is not None:
                    dca += dc
            
            if epoch == 0: continue
            if epoch % parameters['print_every'] == 0:
                print_loss_avg = print_loss_total / parameters['print_every']
                print_loss_total = 0
                print_summary = '%s (%d %d%%) %.4f' % (time_since(start,
                                                                  epoch / float(parameters['n_epochs'])),
                                                       epoch, epoch / parameters['n_epochs'] * 100,
                                                       print_loss_avg)
                print(print_summary)
            
            if epoch % parameters['plot_every'] == 0:
                plot_loss_avg = plot_loss_total / parameters['plot_every']
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
            
            if epoch % parameters['evaluate_every'] == 0:
                for dataset_type in ['train', 'valid', 'test']:
        
                    decoded_word_idxes = all_decoded_word_idxes if dataset_type == 'train' else None
                    print('evaluating on {:s} set ...'.format(dataset_type))
                    current_f1_conll, current_f1_sklearn = evaluate_during_train(dataset_type, parameters,
                                                                                 input_lang, output_lang,
                                                                                 all_pairs[dataset_type],
                                                                                 all_max_len[dataset_type],
                                                                                 all_lengths[dataset_type],
                                                                                 encoder=encoder,
                                                                                 decoder=decoder,
                                                                                 epoch=epoch,
                                                                                 decoded_word_idxes = decoded_word_idxes)
                    f1s_conll[dataset_type]['overall'].append(current_f1_conll['overall'])
                    f1s_sklearn[dataset_type]['overall'].append(current_f1_sklearn['overall'])
                if parameters['main_eval_method'] == 'conll':
                    f1s = f1s_conll
                else:
                    f1s = f1s_sklearn
                current_valid_f1 = f1s['valid']['overall'][epoch-1]
    
                if current_valid_f1 > best_valid_f1:
                    best_valid_f1 = current_valid_f1
                    encoder_path, decoder_path = save_best(encoder, decoder,
                                                           parameters['output_model_filepath'])
                    print("Saved model in {:s}".format(parameters['output_model_filepath']))
                    parameters['encoder_path'] = encoder_path
                    parameters['decoder_path'] = decoder_path
                    patience_cnt = 0
                else:
                    print('The valid F1 does not improve in the last {:d} epochs.'.format(patience_cnt+1))
                    patience_cnt += 1

            if epoch == parameters['n_epochs']:
                if delete_after_train:
                    params = []
                    for m in [encoder, decoder]:
                        if m is not None:
                            params.extend(m.parameters())
                    for obj in params + [encoder, decoder, optimizer, criterion, loss]:
                        delete_obj_if_exist(obj)

            epoch_elapsed_training_time = time.time() - epoch_start_time
            
            print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time))
            if patience_cnt >= parameters['patience']:
                print('Early stopping!')
                break
        
        # finishing up the experiments
        f1s_for_plot = {}
        for dataset_type in ['train', 'valid', 'test']:
            f1s_for_plot[dataset_type] = f1s[dataset_type]['overall']
        graph_path = os.path.join(parameters['output_model_filepath'], 'F1-plot.svg')
        plot_f1(f1s_for_plot, graph_path, 'step'+str(parameters['step']))
        print('F1 (conll):')
        pprint(f1s_conll)
        print('F1 (sklearn):')
        pprint(f1s_sklearn)
        
    except KeyboardInterrupt:
        print("Quitting experiment...")        
        encoder_path, decoder_path = save_best(encoder, decoder,
                                            parameters['output_model_filepath'])
    return best_valid_f1, f1s_conll, f1s_sklearn

def delete_obj_if_exist(obj):
    if obj is not None:
        print('Releasing memory for object...')#.format(obj.__name__))
        # print(obj)
        del obj
    
def evaluate_during_train(dataset_type, parameters, input_lang, output_lang,
                          pairs, max_len, lengths, encoder=None, decoder=None, epoch=0, decoded_word_idxes = None):
    
    if dataset_type == 'train' and decoded_word_idxes is not None:
        # filepaths['lang_obj_file'] = "{:s}obj_{:s}_{:s}".format(data_dir, lang_in, lang_out)
        all_preds_words = np.full((len(pairs), max_len), 's', dtype=object)
        for i in range(len(pairs)):
            for j in range(max_len):
                if int(decoded_word_idxes[i, j]) == 0:
                    continue
                else:
                    word = output_lang.idx2word[int(decoded_word_idxes[i, j])]
                    all_preds_words[i, j] = word
        pred_output_filepath = os.path.join(parameters['pred_output_folder'],
                                            str(epoch) + '.' + dataset_type + '.txt')
        print('writing to {:s}'.format(pred_output_filepath))
        write_pred(all_preds_words, pred_output_filepath, input_lang, output_lang, pairs)
        print('done!')
        current_f1_conll = get_conll_eval(pred_output_filepath)
        current_f1_sklearn = current_f1_conll

    else:
        current_f1_conll, current_f1_sklearn = evaluate(dataset_type, parameters, input_lang, output_lang,
                 pairs, max_len, lengths, encoder, decoder, epoch)

    return current_f1_conll, current_f1_sklearn


def save_model(model, filename):
    if not os.path.isfile(filename):
        open(filename, 'w').close()
    torch.save(model, filename)
    print('Saved %s as %s' % (model.__class__.__name__, filename))

def save_best(encoder, decoder, save_path):
    encoder_path = os.path.join(save_path, 'best-encoder.pt')
    decoder_path = os.path.join(save_path, 'best-decoder.pt')
    save_model(encoder, encoder_path)
    save_model(decoder, decoder_path)
    return encoder_path, decoder_path

def load_model(pretrained_model):
    state_dict = torch.load(pretrained_model)
    new_state_dict = OrderedDict()
    for k, value in state_dict['state_dict'].iteritems():
        key = "module.{}".format(k)
        new_state_dict[key] = value
    torch.load_state_dict(new_state_dict)
    optimizer = state_dict['optimizer']
    return optimizer

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
