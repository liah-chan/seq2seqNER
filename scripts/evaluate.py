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

from data_loader import *
from model import *
import tensor_utils
import utils_data
from subprocess import Popen, PIPE
import re
from sklearn import metrics


use_cuda = torch.cuda.is_available()
# use_cuda = False

def evaluate_step(encoder, decoder, input_lang, output_lang, input_variable, lengths_sorted,
                  input_variable_char=None):
    # this one is for testing BiLSTM
    encoder.eval()
    input_batch_size, input_seq_len = input_variable.size()[1], input_variable.size()[0]
    decoded_words = np.zeros((input_seq_len, input_batch_size), dtype=int)
    encoder_outputs, encoder_hidden, enc_output_lengths = encoder(input_variable,
                                                                  input_lengths = lengths_sorted,
                                                                  input_seqs_char=input_variable_char)
    #encoder_outputs (seq_len, bat_size, out_size)
    #loop over lengths_sorted
    for i_batch in range(input_batch_size):
        seq_len = lengths_sorted[i_batch]
        for i_seq in range(seq_len):
            # break to see the prediction output is one int or a vec
            topv, topi = encoder_outputs.data.topk(1, dim=-1)
            decoded_word = topi[i_seq, i_batch].cpu().numpy()
            decoded_words[i_seq, i_batch] = decoded_word

    decoded_words = np.transpose(decoded_words, (1, 0))
    return decoded_words, None

def evaluate_step_seq2seq(encoder, decoder, input_lang, output_lang, input_variable, lengths_sorted,
                          input_variable_char=None):
    input_batch_size, input_seq_len = input_variable.size()[1], input_variable.size()[0]

    encoder.eval()
    decoder.eval()

    decoded_words = np.zeros((input_seq_len, input_batch_size), dtype=int)
    #TODO: does the encoder_hidden need to be initialized here?
    encoder_outputs, encoder_hidden, enc_output_lengths = encoder(input_variable,
                                                            input_lengths = lengths_sorted,
                                                            input_seqs_char=input_variable_char)

    decoder_input = Variable(torch.LongTensor([input_lang.word2idx['<SOS>']] * input_batch_size).unsqueeze(0))

    decoder_hidden = (encoder_hidden[0][1].unsqueeze(0), encoder_hidden[1][1].unsqueeze(0))

    if use_cuda:
        decoder_input = decoder_input.cuda()

    decoder_attentions = torch.zeros(input_batch_size, input_seq_len , input_seq_len )

    for di in range(input_seq_len):
        decoder_output, decoder_hidden, decoder_attention = decoder(di,
            decoder_input, decoder_hidden, encoder_outputs, input_lengths=lengths_sorted)

        _, topi = decoder_output.data.topk(2, dim=-1)
        decoded_word = topi[:, :, 0]
        decoded_words[di,:] = decoded_word.cpu().numpy()
        decoder_attentions[:, di, :decoder_attention.size(2)] = decoder_attention.squeeze(1).cpu().data
        decoder_input = Variable(decoded_word)
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        PAD_tokens = torch.from_numpy(np.zeros((decoded_word.size(0), decoded_word.size(1)))).type(torch.LongTensor)
        PAD_tokens = PAD_tokens.cuda() if use_cuda else PAD_tokens

        wrong_stops = torch.eq(decoded_word, PAD_tokens).type(torch.FloatTensor)
        if wrong_stops.sum() > 1 and di != input_seq_len - 1:
            indexes = wrong_stops.nonzero()[:,1].numpy().tolist()
            print('indexes.size()')
            print(len(indexes))
            print('indexes:')
            print(indexes)
            for index in indexes:
                decoded_word[0, index] = topi[0,index,1]
                decoder_input = Variable(decoded_word)
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            decoded_words[di,:] = decoded_word.cpu().numpy()
        # decoded_words[:, di] = ni.cpu().numpy().reshape(-1)
    decoded_words = np.transpose(decoded_words, (1, 0))

    return decoded_words, decoder_attentions


def write_pred(preds, predfile, input_lang, output_lang, pairs):
    '''
    inputs and targets are torch tensors, need to be converted back to words
    preds are already words
    '''
    try:
        assert len(pairs) == preds.shape[0]
    except AssertionError:
        print('evaluation pairs shape and preds shape are not the same!!! got:')
        print('len(pairs)', len(pairs), 'preds.shape[0]',preds.shape[0])
    with open(predfile, 'w') as f:
        preds = preds.tolist()
        
        for i in range(len(pairs)):
            input_seq = pairs[i][0].split(' ')
            target_seq = pairs[i][1].split(' ')
            pred_seq = preds[i][:len(target_seq)]
            assert len(input_seq) == len(target_seq) and len(target_seq) == len(pred_seq)
            for i in range(len(input_seq)):
                line = input_seq[i] + ' ' + target_seq[i] + ' ' + pred_seq[i] + '\n'
                f.writelines(line)
            f.writelines(u'\n')

def evaluate(dataset_type,parameters, input_lang, output_lang, pairs, max_len,
											lengths, encoder, decoder, epoch):
    parameters['dataset_type'] = dataset_type
    if parameters['pred_output_folder'] is None:
        print('has to provide a output folder for prediction results')
        sys.exit(0)
    if parameters['conll_filepath'] is None:
        print('has to provide conll file path')
        sys.exit(0)
    
    encoder = torch.load(parameters['encoder_path']) if encoder is None else encoder
    if parameters['nn_model'] == 'seq2seq':
        decoder = torch.load(parameters['decoder_path']) if decoder is None else decoder
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every
    
    all_preds = np.zeros((len(pairs), max_len), dtype=int)
    # all_golds = np.zeros((len(pairs), max_len))
    all_preds_list = []
    all_golds_list = []
    all_preds_words = np.full((len(pairs), max_len), 's', dtype = object)
    
    for i_batch, ((input_variables, target_variables, input_variables_char), batch_lengths) in \
     enumerate(get_batch(input_lang, output_lang, pairs, \
     batch_size = parameters['batch_size'], max_len =max_len, lengths = lengths)):
    
        input_variables_sorted, input_variables_char_sorted, target_variables_sorted, \
        lengths_sorted, lengths_argsort \
            = tensor_utils.sort_variables_lengths(input_variables, target_variables,
                                                  batch_lengths, needs_argsort=True,
                                                  input_variables_char=input_variables_char, 
                                                  if_volatile = True)

        if parameters['nn_model'] == 'seq2seq':
            preds, attns = evaluate_step_seq2seq(encoder, decoder, input_lang, output_lang,
                              input_variables_sorted, lengths_sorted,
                              input_variable_char=input_variables_char_sorted)
        elif parameters['nn_model'] == 'bilstm':
            preds, attns = evaluate_step(encoder, decoder, input_lang, output_lang,
                              input_variables_sorted, lengths_sorted,
                              input_variable_char=input_variables_char_sorted)
        # here the preds in each batch are sorted accroding to sequence length
        new_preds = tensor_utils.sort_results_back(preds, lengths_argsort)
        all_preds[i_batch * parameters['batch_size'] : \
                    min((i_batch + 1) * parameters['batch_size'],len(pairs)), :] = new_preds
        # all_golds[i_batch * parameters['batch_size'] : \
        #             min((i_batch + 1) * parameters['batch_size'],len(pairs)), :] = np.array(target_variables)
        all_preds_list.extend([item for sublist in new_preds.tolist() for item in sublist])
        all_golds_list.extend([item for sublist in target_variables for item in sublist])

    ### calc sklearn F1
    new_y_pred, new_y_true, \
    new_label_indices, new_label_names, _, _ = utils_data.remap_labels(all_preds_list,
                                                                       all_golds_list,
                                                                       output_lang,
                                                                       'token')
    ix = new_label_names.index('<SOS>')
    new_label_names.remove('<SOS>')
    new_label_indices.remove(new_label_indices[ix])
    current_f1_report = metrics.classification_report(y_pred=new_y_pred, y_true=new_y_true,
                                                    digits=4,
                                                    labels=new_label_indices,
                                                    target_names=new_label_names)
    
    current_f1_sklearn = get_sklearn_eval(current_f1_report,dataset_type)

    for i in range(len(pairs)):
        for j in range(max_len):
            if int(all_preds[i,j]) == 0:
                continue
            else:
                word = output_lang.idx2word[int(all_preds[i,j])]
                all_preds_words[i,j] = word
    pred_output_filepath = os.path.join(parameters['pred_output_folder'],
                                    str(epoch)+'.'+parameters['dataset_type']+'.txt' )
    print('writing pred results to {:s}'.format(pred_output_filepath))
    write_pred(all_preds_words, pred_output_filepath, input_lang, output_lang, pairs)
    print('done!')
    current_f1_conll = get_conll_eval(pred_output_filepath)
    
    return current_f1_conll, current_f1_sklearn

def get_sklearn_eval(current_f1_report, dataset_type):
    print('F1 (sklearn) on {:s} set:'.format(dataset_type))
    print(current_f1_report)
    f1_scores = {}
    pattern = r'(\w+)\s+\d\.\d+\s+\d\.\d+\s+(\d\.\d+)'
    for line in current_f1_report.strip().split('\n'):
        match = re.search(pattern, line)
        if match:
            if match.group(1) == 'total':
                f1_scores['overall'] = float(match.group(2)) * 100.
            else:
                f1_scores[match.group(1)] = float(match.group(2)) * 100.
    return f1_scores


def get_conll_eval(pred_output_filepath):
    # subprocess.call('perl conlleval <{}'.format(pred_output_filepath), shell = True)
    with open(pred_output_filepath, 'r') as pred_output_file:
        p = Popen(['perl', 'conlleval'], stdin=pred_output_file, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    # rc = p.returncode
    print(output)
    output_lines = output.split('\n')
    pattern = r'(\w+).*FB1:\s*(\d+.\d+)'
    f1_scores = {}
    for line in output_lines:
        match = re.search(pattern=pattern, string=line)
        if match:
            if match.group(1) == 'accuracy':
                f1_scores['overall'] = float(match.group(2))
            else:
                f1_scores[match.group(1)] = float(match.group(2))
    pprint(f1_scores)
    return f1_scores
