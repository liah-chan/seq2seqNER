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

def sort_variables_lengths(input_variables, target_variables, batch_lengths, needs_argsort = False,
						   input_variables_char = None, if_volatile = False):
	'''
	sorting input/target batch according to the sequence length, descending order
	'''
	input_variables_sorted, target_variables_sorted, lengths_sorted = [], [], []
	if input_variables_char is not None:
		input_variables_char_sorted = []
	# the indexes of the variables to be sorted, from longest to the shortest
	lengths_argsort = sorted(range(len(batch_lengths)), 
							key=lambda k: batch_lengths[k], 
							reverse = True)
	for i in lengths_argsort:
		input_variables_sorted.append(input_variables[i])
		target_variables_sorted.append(target_variables[i])
		lengths_sorted.append(batch_lengths[i])
		if input_variables_char is not None:
			input_variables_char_sorted.append(input_variables_char[i])

	input_variables_sorted = Variable(torch.LongTensor(input_variables_sorted).transpose(0,1), volatile = if_volatile)
	target_variables_sorted = Variable(torch.LongTensor(target_variables_sorted).transpose(0,1), volatile = if_volatile)
			
	if input_variables_char is not None:
		input_variables_char_sorted = Variable(torch.LongTensor(input_variables_char_sorted).transpose(0,1))
		if use_cuda:
			input_variables_char_sorted = input_variables_char_sorted.cuda()
			input_variables_sorted = input_variables_sorted.cuda()
			target_variables_sorted = target_variables_sorted.cuda()
	
	if needs_argsort:
		return input_variables_sorted, input_variables_char_sorted,\
			   target_variables_sorted, lengths_sorted, lengths_argsort
	else:
		return input_variables_sorted, input_variables_char_sorted,\
			   target_variables_sorted, lengths_sorted

def sort_results_back(results, argsort, return_list = False):
	new_results = np.zeros_like(results)
	c = 0
	for i in argsort:
		new_results[i] = results[c]
		c += 1
	return new_results