import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(2 * self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if use_cuda:
            attn_energies = attn_energies.cuda()

        attn_energies = self.score(hidden, encoder_outputs, this_batch_size, max_len)

        normalized_energies = F.softmax(attn_energies.transpose(0,2)).transpose(0,2)
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        # return F.softmax(attn_energies)#.unsqueeze(1)
        return normalized_energies
  
    def score(self, hidden, encoder_output, batch_size, seq_len):
        hidden = hidden.transpose(0,1)
        if self.method == 'general':
            encoder_output = encoder_output.contiguous().view(batch_size * seq_len, -1)
            energy = self.attn(encoder_output).view(batch_size , seq_len, -1)
            energy = energy.transpose(2, 1)
            energy = hidden.bmm(energy)
            return energy
        
        elif self.method == 'concat':
            # broadcast hidden to encoder_outputs size
            hidden = hidden * Variable(torch.ones(encoder_output.size())) 
            energy = self.attn(torch.cat((hidden, encoder_output), -1))
            energy = energy.transpose(2, 1)
            energy = self.v.bmm(energy)
            return energy
        # elif self.attn_mode == 'concat':
        elif self.method == 'dot':
            encoder_output = encoder_output.transpose(2, 1)
            energy = hidden.bmm(encoder_output)
            return energy
        else:
            raise ValueError('attention mode not recognized')