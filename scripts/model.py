# import mkl
# mkl.set_num_threads(16)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from attn import *

use_cuda = torch.cuda.is_available()

class BiLSTM(nn.Module):
# class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size,
                 n_layers=1, dropout=0.1, pretrained_weights=None, output_size = 10,
                 use_char_embedding = False, char_alphabet_size = 0, char_emb_size = 25):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.use_char_embedding = use_char_embedding

        self.embedding = nn.Embedding(input_size, emb_size)
        if pretrained_weights is not None:
            weights = torch.from_numpy(pretrained_weights).type(torch.FloatTensor)
            assert weights.size(0) == input_size and weights.size(1) == emb_size
            weights = weights.cuda() if use_cuda else weights
            self.embedding.weight = nn.Parameter(weights)
        
        ### added char embedding
        if self.use_char_embedding and char_alphabet_size != 0:
            self.char_alphabet_size = char_alphabet_size
            self.char_emb_size = char_emb_size
            self.char_embedding = nn.Embedding(self.char_alphabet_size, self.char_emb_size)
            
            self.char_lstm = nn.LSTM(char_emb_size, char_emb_size,
                                     n_layers, dropout = 0.1,
                                     bidirectional=True)
            
        self.dropout = nn.Dropout(p=self.dropout_p)
        if not self.use_char_embedding:
            self.lstm = nn.LSTM(emb_size, hidden_size,
                            n_layers, dropout=0.1,
                            bidirectional=True)
        else:
            self.lstm = nn.LSTM(emb_size + char_emb_size * 2, hidden_size,
                                n_layers, dropout=0.1,
                                bidirectional=True)

        self.output_size = output_size
        self.out1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.out2 = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_seqs, input_lengths, hidden=None, input_seqs_char = None):
        embedded = self.embedding(input_seqs)
        embedded = self.dropout(embedded)
        
        if self.use_char_embedding and input_seqs_char is not None:
            char_batch_size = input_seqs_char.size(1)
            char_seq_len = input_seqs_char.size(0)
            input_seqs_char = input_seqs_char.view(char_seq_len * char_batch_size, -1)
            char_embedded = self.char_embedding(input_seqs_char).transpose(1, 0)
            char_lstm_output, (char_lstm_hidden, char_lstm_cell) = self.char_lstm(char_embedded)
            char_lstm_output_sum = torch.cat((char_lstm_hidden[0], char_lstm_hidden[1]), -1)
            char_embedded_seq = char_lstm_output_sum.view(char_seq_len, char_batch_size, -1)
            embedded = torch.cat((embedded, char_embedded_seq), -1)
                
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        seq_len = outputs.size(0)
        batch_size = outputs.size(1)
        raw_outputs1 = self.out1(outputs.view(seq_len * batch_size, -1))
        raw_outputs2 = self.out2(raw_outputs1).view(seq_len, batch_size, -1)
        scores = self.softmax(raw_outputs2.transpose(2, 0))
        scores = scores.transpose(2, 0)

        return scores, hidden, output_lengths
    
    def init_hidden(self, batch_size):
        c_state = Variable(torch.randn(self.n_layers * 2, batch_size, self.hidden_size))
        h_state = Variable(torch.randn(self.n_layers * 2, batch_size, self.hidden_size))
        if use_cuda:
            c_state = c_state.cuda()
            h_state = h_state.cuda()
        return (c_state, h_state)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size,  # alphabet_size, char_emb_size,
                 n_layers=1, dropout=0.1, pretrained_weights=None,
                 use_char_embedding=False, char_alphabet_size=0, char_emb_size=25):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.use_char_embedding = use_char_embedding
        
        self.embedding = nn.Embedding(input_size, emb_size)
        if pretrained_weights is not None:
            weights = torch.from_numpy(pretrained_weights).type(torch.FloatTensor)
            assert weights.size(0) == input_size and weights.size(1) == emb_size
            weights = weights.cuda() if use_cuda else weights
            self.embedding.weight = nn.Parameter(weights)
        
        ### added char embedding
        if self.use_char_embedding and char_alphabet_size != 0:
            self.char_alphabet_size = char_alphabet_size
            self.char_emb_size = char_emb_size
            self.char_embedding = nn.Embedding(self.char_alphabet_size, self.char_emb_size)
            
            self.char_lstm = nn.LSTM(char_emb_size, char_emb_size,
                                     n_layers,
                                     bidirectional=True)
        
        self.dropout = nn.Dropout(p=self.dropout_p)
        if not self.use_char_embedding:
            self.lstm = nn.LSTM(emb_size, hidden_size,
                                n_layers,
                                bidirectional=True)
        else:
            self.lstm = nn.LSTM(emb_size + char_emb_size * 2, hidden_size,
                                n_layers,
                                bidirectional=True)
        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, input_seqs, input_lengths, hidden=None, input_seqs_char=None):
        
        embedded = self.embedding(input_seqs)
        embedded = self.dropout(embedded)
        if self.use_char_embedding and input_seqs_char is not None:
            char_batch_size = input_seqs_char.size(1)
            char_seq_len = input_seqs_char.size(0)
            input_seqs_char = input_seqs_char.contiguous().view(char_seq_len * char_batch_size, -1)
            char_embedded = self.char_embedding(input_seqs_char).transpose(1, 0)
            char_lstm_output, (char_lstm_hidden, char_lstm_cell) = self.char_lstm(char_embedded)
            char_lstm_output_sum = torch.cat((char_lstm_hidden[0], char_lstm_hidden[1]), -1)
            char_embedded_seq = char_lstm_output_sum.view(char_seq_len, char_batch_size, -1)
            embedded = torch.cat((embedded, char_embedded_seq), -1)
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, encoder_hidden = self.lstm(packed, hidden)
        outputs, enc_output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # the output passed into the decoder should be a concatenation, twice of hidden size
        encoder_outputs = torch.cat((outputs[:, :, :self.hidden_size], outputs[:, :, self.hidden_size:]), -1)  # concat bidirectional outputs
        
        return encoder_outputs, encoder_hidden, enc_output_lengths
    
    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        c_state = Variable(torch.randn(self.n_layers * 2, batch_size, self.hidden_size))
        h_state = Variable(torch.randn(self.n_layers * 2, batch_size, self.hidden_size))
        if use_cuda:
            c_state = c_state.cuda()
            h_state = h_state.cuda()
        return (c_state, h_state)


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # self.max_length = max_length
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('general', hidden_size)
        self.lstm = nn.LSTM(5 * hidden_size, hidden_size, n_layers)
        self.out1 = nn.Linear(3 * hidden_size, hidden_size)
        self.out2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, di, word_input, last_hidden, encoder_outputs, input_lengths):
        this_batch_size = word_input.size()[1]
        # print('In decoder: this_batch_size ', this_batch_size) #100L
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, this_batch_size, -1)  # S=1 x B x N
        word_embedded = self.dropout(word_embedded)        
        attn_weights = self.attn(last_hidden[0], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        context = context.transpose(0, 1)  # 1 x B x N
        
        if di < len(encoder_outputs):
            aligned_encoder_hidden = encoder_outputs[di, :, :].unsqueeze(0)
        else:
            # TODO Find a proper aligned_encoder_hidden in the case that the seq end before max_len
            aligned_encoder_hidden = encoder_outputs[len(encoder_outputs) - 1, :, :].unsqueeze(0)
        rnn_input = torch.cat((word_embedded, context, aligned_encoder_hidden), 2)
        packed_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, input_lengths)
        output, hidden = self.lstm(packed_rnn_input, last_hidden)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output)  # unpack (back to padded)
        
        input_to_final = torch.cat((output, context), -1)
        final_batch_size = input_to_final.size()[1]
        final_len = input_to_final.size()[0]
        input_to_final = input_to_final.view(final_batch_size * final_len, -1)
        final_outputs1 = self.out1(input_to_final)
        final_outputs2 = self.out2(final_outputs1).view(final_len, final_batch_size, -1)
        # output = F.log_softmax(self.out(input_to_final))  # ???? why
        scores = self.softmax(final_outputs2.transpose(2, 0))
        scores = scores.transpose(2, 0)
        
        return scores, hidden, attn_weights