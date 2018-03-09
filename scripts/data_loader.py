from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle
import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import random
import numpy as np

random.seed(5678)
use_cuda = torch.cuda.is_available()
# use_cuda = False
class Lang:
    def __init__(self, name, is_output):
        self.name = name
        self.is_output = is_output
        if is_output:
            self.word2idx = {'<SOS>':1}
            self.idx2word = {1: '<SOS>'}
            self.words = ['<SOS>']
            self.n_words = 1

        else:
            #save 0 for padding
            self.word2idx = {'<SOS>':1, '<EOS>':2, '<UNK>': 3}
            self.idx2word = {1:'<SOS>',2:'<EOS>', 3: '<UNK>'}
            self.words = ['<SOS>', '<EOS>', '<UNK>']
            self.n_words = 3
            
            # creating alphabet dict for character embedding
            self.char2idx = {}
            self.idx2char = {}
            self.chars = []
            self.n_chars = 0
            self.max_word_len = 0
            self.max_word = ''
        
        self.word_count = {}
        self.unk_list = []
        self.char_count = {}
        
    def add_sentence(self, sentence):
        for word in sentence.split(' '):
                self.add_word(word)

    def add_word(self, word):
        if word not in self.words:
            self.words.append(word)
            self.word_count[word] = 1
            self.n_words += 1

        else:
            self.word_count[word] += 1
        
        if not self.is_output:
            word_len = 0
            for char in word:
                self.add_char(char)
                word_len+=1
                if word_len > self.max_word_len:
                    self.max_word_len = word_len
                    self.max_word = word
                    
    def add_char(self, char):
        if char not in self.chars:
            self.chars.append(char)
            self.char_count[char] = 1
            self.n_chars += 1
        else:
            self.char_count[char] += 1
            
    def make_char_dict(self):
        for idx, char in enumerate(self.chars):
            self.char2idx[char] = idx + 1
            self.idx2char[idx+1] = char
            
    def make_word_dict(self, is_output = False, shuffle = True):
        
        if shuffle:
            if is_output:
                preserved_tags = self.words[:1]
                actual_words = self.words[1:]
                random.shuffle(actual_words)
                self.words = preserved_tags + actual_words
                
                # random.shuffle(self.words)

            else:
                preserved_tags = self.words[:3]
                actual_words = self.words[3:]
                random.shuffle(actual_words)
                self.words = preserved_tags + actual_words
                
        for idx, word in enumerate(self.words):
            self.word2idx[word] = idx + 1
            self.idx2word[idx+1] = word
        # if not is_output:
    
    # def find_max_word_len(self):
    #     for word_cnt
        
# turn Unicode characters to ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    # s = unicodeToAscii(s.lower().strip())
    s = unicodeToAscii(s.strip())
    # substituting digits with 0, but keep the format
    s = re.sub(r"\d", r"0", s)
    return s

def readLangs(lang1, lang2, dataset_filepath, lang_obj_file, reverse = False, exp_step = 1, resume_training = False):
    """
    reading data (train, valid and test) file, spliting to lines and trans pairs
    return:
    all_pairs: dictionary ['train']['valid']['test'] containing lists of sentence and label sequence pairs.
               e.g.: [u'SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .',
                      u'O O O O O O O B-PER O O O O']
               
    """
    # normal step1
    if exp_step == 1 and resume_training == False:
        print('readLangs in step 1.\nInitiating Lang objects...')
        input_lang = Lang(lang1, is_output=False)
        output_lang = Lang(lang2, is_output=True)
    
    # baseline
    elif exp_step == 2 and resume_training == False:
        print('readLangs in for baseline step.\nLoading Lang objects... ')
        (input_lang, output_lang, _) = load_lang_from_file(lang_obj_file)

    # both step 1 and step 2, resume training from a checkpoint
    elif exp_step in [1,2] and resume_training == True:
        print('readLangs in step 2.\nLoading Lang objects... ')
        (input_lang, output_lang, _) = load_lang_from_file(lang_obj_file)
        # sys.exit(0)

    all_pairs = {}
    for data_type in ['train', 'valid', 'test']:
        data_filepath = os.path.join(dataset_filepath, data_type+'.txt')
        print('reading lines in {:s} file (path: {:s})'.format(data_type, data_filepath))
        pairs = []
        for line in open(data_filepath, encoding = 'utf-8'):
            pairs.append([normalizeString(s) for s in line.strip().split('\t')])
        all_pairs[data_type] = pairs
        print('Read {:d} pairs in {:s} set'.format(len(all_pairs[data_type]), data_type))

    return input_lang, output_lang, all_pairs

def prepareData(lang1, lang2, dataset_filepath, data_type, lang_obj_file, reverse=False,
                exp_step = 1, add_class = False, resume_training = False):
    '''
    Prepare for train, valid and test
    :param lang1:
    :param lang2:
    :param dataset_filepath:
    :param data_type:
    :param lang_obj_file:
    :param reverse:
    :return:
    '''
    input_lang, output_lang, all_pairs = readLangs(lang1, lang2, dataset_filepath,
                                               lang_obj_file, reverse, exp_step, resume_training)
    # train_pairs, val_pair, test_pairs = pairs
    all_lengths = {}
    all_max_len = {}
    for data_type, pairs in all_pairs.iteritems():
        all_lengths[data_type] = [len(pairs[x][0].strip().split(' ')) for x in range(len(pairs))]
        all_max_len[data_type] = max(all_lengths[data_type])
        print('Counting sentence length in {:s} set...'.format(data_type))
        print("For {:s} set, maximum length of sentence sentence : {:d}".format(data_type, all_max_len[data_type]))

    for data_type in all_pairs.keys():
        if exp_step == 1 and not resume_training:
            print('Processing {:s} set, adding words to the dictionary...'.format(data_type))
            for pair in all_pairs[data_type]:
                input_lang.add_sentence(pair[0])
                output_lang.add_sentence(pair[1])
            # break here to see the input_lang.chars char2idx idx2char
            print('shuffle word dictionary...')
            input_lang.make_word_dict(is_output=False)
            input_lang.make_char_dict()
            output_lang.make_word_dict(is_output=True)
            print('shuffle char dictionary...')
            input_lang.make_char_dict()
            print('Done!')

        else:
            ### TODO: need to verify whether only the make_word_dict is not needed
            print('In step 2, not adding any words to Lang objects.')
            # unk_idx = input_lang.word2idx['<UNK>']
            if add_class:
                for pair in all_pairs[data_type]:
                    output_lang.add_sentence(pair[1])
                # make sure new labels are added to the end of index
                for word in output_lang.words:
                    if word not in output_lang.word2idx:
                        idx = len(output_lang.word2idx.keys())
                        output_lang.word2idx[word] = idx + 1
                        output_lang.idx2word[idx + 1] = word
                # TODO: the new output_lang object is not saved
    
    if exp_step == 1 and not resume_training:
        print('In step 1, saving Lang objects to file {:s}'.format(lang_obj_file))
        save_lang_to_file(input_lang, output_lang, all_pairs, lang_obj_file)

    print("Counted vocab size:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print(output_lang.idx2word)
    print('idx2word for input_lang:')
    print([input_lang.idx2word[i] for i in range(1, 31)])
    print('idx2char for input_lang:')
    print([input_lang.idx2char[i] for i in range(1,31)])
    return input_lang, output_lang, all_pairs, all_max_len, all_lengths

def variables_from_pairs(input_lang, output_lang, pairs, max_len):
    x_variables = []
    y_variables = []
    x_variables_char = []
    # unk_idx = input_lang.word2idx['<UNK>']
    for pair in pairs:
        x_variable, y_variable, x_variable_char = variables_from_pair(input_lang, output_lang,
                                                                      pair, max_len)
        x_variables.append(x_variable)
        y_variables.append(y_variable)
        x_variables_char.append(x_variable_char)
    return x_variables, y_variables , x_variables_char

#get a pair of input/output, make them into a tuple of torch Var

def variables_from_pair(input_lang, output_lang, pair, max_len):
    '''
    :param input_lang: Lang object
    :param output_lang: Lang object
    :param pair:
    :param max_len:
    :return:
    '''
    
    # unk_idx = input_lang.word2idx['<UNK>']
    input_variable = variable_from_sentence(input_lang, pair[0], max_len)
    target_variable = variable_from_sentence(output_lang, pair[1], max_len)
    input_variable_char = []
    cnt = 0
    #getting char var:
    for word in pair[0].split(' '):
        input_variable_char.append(variable_from_word(input_lang, word, input_lang.max_word_len))
        cnt += 1
    # new_var = Variable(torch.LongTensor([0] * input_lang.max_word_len).view(-1, 1))
    # new_var = new_var.cuda() if use_cuda else new_var
    new_var = [0] * input_lang.max_word_len
    for i in range(max_len - cnt): #make input_variable_char the length of max_len
        input_variable_char.append(new_var)
        # input_variable_char should be list of list
    return (input_variable, target_variable, input_variable_char)

def variable_from_word(lang, word, max_word_len):
    '''
    :param lang: Lang object
    :param word: str, a word
    :param max_word_len: int, max len of a word
    :return indexes: int list, length: max_len
    '''
    indexes = indexes_from_word(lang, word)
    if len(indexes) < max_word_len:
        indexes = indexes + [0] * (max_word_len - len(indexes))
    # result = Variable(torch.LongTensor(indexes).view(-1, 1))  # make it into shape [sent_len+1, 1]
    # if use_cuda:
    #     return result.cuda()
    # else:
    return indexes
    
def variable_from_sentence(lang, sentence, max_len):
    '''
    :param lang: Lang object
    :param sentence: str, a sentence, each word seperated by ' '
    :param max_len: int, max len of the sentence (i.e. max len of the returned list),
                    sentences which has less words than max len are padded with 0 in the end.
    :return result: pytorch Longtensor variable, length: max_len
    '''
    
    indexes = indexes_from_sentence(lang, sentence) # a sentence as indices.
    # indexes.append(EOS_token_idx)
    #add zero paddings to the end
    if len(indexes) < max_len:
    	indexes = indexes + [0] * (max_len - len(indexes))
    return indexes

def indexes_from_word(lang, word):
    '''
    :param lang: Lang object
    :param word: str, a word
    :return idxes: int list, each char in the word mapped to indexes
    '''
    idxes = []
    for char in word:
        idx = lang.char2idx[char]
        idxes.append(idx)
    return idxes

def indexes_from_sentence(lang, sentence):
    '''
    :param lang: Lang object
    :param sentence: str, a sentence, each word seperated by ' '
    :return idxes: int list, sentence mapped to indexes
    '''
    idxes = []
    for word in sentence.split(' '):
        try:
            idx = lang.word2idx[word]
        except KeyError:
            try:
                idx = lang.word2idx['<UNK>']
                lang.unk_list.append(word)
            except KeyError:
                print('keyerror: {:s}'.format(word))
        idxes.append(idx)
    return idxes
    # return [lang.word2idx[word] for word in sentence.split(' ')]

def save_lang_to_file(input_lang, output_lang, pairs, fname):
    with open(fname, 'wb') as f:
        pickle.dump((input_lang,output_lang,pairs), f)
    print('saved lang object to {:s}'.format(fname))

def load_lang_from_file(fname):
    with open(fname, 'rb') as f:
        data= pickle.load(f)
    return data

def get_batch(input_lang, output_lang, pairs, batch_size, max_len, lengths):
    # unk_idx = input_lang.word2idx['<UNK>']
    data_size = len(pairs)
    num_batches = int((data_size - 1)/batch_size) + 1
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield variables_from_pairs(input_lang, output_lang, pairs[start_index:end_index], max_len), lengths[start_index:end_index]

def sentences_from_variables(lang, variables):
    sentences = []
    # print(variables[0])
    # print(len(variables))
    for i in range(len(variables)):
        sentence = sentence_from_variable(lang, variables[i])
        sentences.append(sentence)
    return sentences

def sentence_from_variable(lang, variable):
    indexes = variable.cpu().data.numpy().reshape(-1).tolist()
    # print(indexes)
    words = []
    for idx in indexes:
        if idx in [0]:
            continue
        else:
            word = lang.idx2word[idx]
            words.append(word)
    return words

def load_pretrained_word_embeddings(parameters, input_lang):
    emb_file = parameters['embedding_filepath']
    f = open(emb_file, 'r')
    line = f.readline()
    emb_size = len(line.strip().split(' ')) - 1
    f.close()
    print('loading word embeddings from file {:s}...\n Embedding size: {:d}'.format(
        emb_file, emb_size
    ))
    
    embbedding_weights = np.zeros((input_lang.n_words + 1, emb_size))
    pretrained_embeddings = {}
    with open(emb_file, 'r') as f:
        # found_cnt = 0
        # mean = 0
        # stddev = 0
        for line in f:
            splited = line.strip().split(' ')
            if len(splited) == 0:
                continue
            else:
                pretrained_embeddings[splited[0]] = splited[1:]
    direct_map = 0
    lowercase_map = 0
    random_init = 0
    map_to_unk = 0
    low_frequency_word = []
    others = []
    words_without_pretrained_embeddings = []
    for word in input_lang.words:
        if word in ['<SOS>', '<EOS>', '<UNK>']:
            continue
        elif word in pretrained_embeddings:
            vector = np.array(pretrained_embeddings[word], dtype=float)
            embbedding_weights[input_lang.word2idx[word]] = vector
            direct_map += 1
        elif word.lower() in pretrained_embeddings:
            vector = np.array(pretrained_embeddings[word.lower()], dtype=float)
            embbedding_weights[input_lang.word2idx[word]] = vector
            lowercase_map += 1
        elif input_lang.word_count[word] > 1:
            # not low frequency word, but in
            #random init
            vector = np.random.uniform(-0.25, 0.25, emb_size)
            embbedding_weights[input_lang.word2idx[word]] = vector
            random_init += 1
        elif input_lang.word_count[word] <= 1:
            low_frequency_word.append(word)
        else:
            others.append(word)
    
    print('Map {:d} tokens with pretrained embeddings.'.format(direct_map+lowercase_map))
    print('direct map: {:d}\nlower-case map: {:d}\n'.format(direct_map, lowercase_map))
    print('Randomly initialized {:d} token embeddings.'.format(random_init))
    print('{:d} low_frequency_word: '.format(len(low_frequency_word)), low_frequency_word)
    
    return embbedding_weights, emb_size