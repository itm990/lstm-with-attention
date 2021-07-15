# Input-feeding Approach
# 実装間違い

import torch
import torch.nn as nn

# Encoder の定義
class EncoderLSTM(nn.Module):
    def __init__(self, PAD, hidden_size, dict_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(dict_size, hidden_size, padding_idx=PAD)
        self.dropout = nn.Dropout(p=0.1)
        self.lstmcell = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, input_, pre_hidden, pre_cell):
        # batch_size --> (batch_size * hidden_size)
        embedded = self.embedding(input_)
        embedded = self.dropout(embedded)
        hidden, cell = self.lstmcell(embedded, (pre_hidden, pre_cell))

        # masking
        mask = torch.unsqueeze(input_, dim=1).repeat(1, self.hidden_size)
        hidden = torch.where(mask==0, pre_hidden, hidden)
        cell = torch.where(mask==0, pre_cell, cell)
        
        return hidden, cell

# Decoder の定義
class AttentionDecoderLSTM(nn.Module):
    def __init__(self, PAD, hidden_size, dict_size):
        super(AttentionDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(dict_size, hidden_size, padding_idx=PAD)
        self.dropout = nn.Dropout(p=0.1)
        self.lstmcell = nn.LSTMCell(hidden_size, hidden_size)

        self.align_softmax = nn.Softmax(dim=2)
        self.hidden_out = nn.Linear(hidden_size*2, hidden_size)
        self.tanh = nn.Tanh()
        
        self.out = nn.Linear(hidden_size, dict_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input_, pre_hidden, pre_cell, source_hiddens, source_inputs):
        # batch_size --> (batch_size * hidden_size)
        embedded = self.embedding(input_)
        embedded = self.dropout(embedded)
        hidden, cell = self.lstmcell(embedded, (pre_hidden, pre_cell))

        # masking
        mask = torch.t(input_.repeat(self.hidden_size, 1))
        hidden = torch.where(mask==0, pre_hidden, hidden)
        cell = torch.where(mask==0, pre_cell, cell)


        # source_hiddens (batch_size * sentence_length * hidden_size)

        # scores, alignments (batch_size * 1 * sentence_length)
        scores = torch.bmm(torch.unsqueeze(hidden, dim=1), torch.transpose(source_hiddens, dim0=1, dim1=2)) 
        attn_mask = torch.unsqueeze(source_inputs, dim=1)
        infs = torch.full_like(scores, -float('inf'))
        scores = torch.where(attn_mask==0, infs, scores)
        alignments = self.align_softmax(scores)

        # context (batch_size * 1 * hidden_size) --> (batch_size * hidden_size)
        context = torch.bmm(alignments, source_hiddens)
        context = torch.squeeze(context, dim=1)
        
        # attn_hidden (batch_size * hidden_size*2) --> (batch_size * hidden_size)
        attn_hidden = torch.cat((context, hidden), dim=1)
        attn_hidden = self.hidden_out(attn_hidden)
        attn_hidden = self.tanh(attn_hidden)

        
        output = self.out(attn_hidden)
        output = self.softmax(output)

        return output, attn_hidden, cell
