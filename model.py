import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):

        outputs, hidden = self.gru(src, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn1 = nn.Linear(self.hidden_size * 2, hidden_size)
        self.attn2 = nn.Linear(hidden_size, 1) 
    
    def _init_hidden(self):
        nn.init.xavier_normal_(self.attn1.weight)
        nn.init.xavier_normal_(self.attn2.weight)

    def forward(self, hidden, encoder_outputs): 
        seq_len, batch_size, _ = encoder_outputs.size()
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        inputs = torch.cat((encoder_outputs, h), 2).view(-1, self.hidden_size*2)
        o = self.attn2(F.tanh(self.attn1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs)
        # print('context size is:', context.size())>>>(16, 1, 1024)
        return context

    # def score(self, hidden, encoder_outputs):
    #     # [B*T*2H]->[B*T*H]
    #     assert(hidden.size(2) + encoder_outputs.size(2) == self.hidden_size*2)
    #     energy = self.attn(torch.cat([hidden, encoder_outputs], 2)).softmax(-1)
    #     energy = energy.transpose(1, 2)  # [B*H*T]
    #     v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
    #     #print('')
    #     energy = torch.bmm(v, energy)  # [B*1*T]
    #     return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        #print('input size is:', input.size())
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        # print('last_hidden size is:', last_hidden.size())>>>[3, 16, (1024)hidden_size]
        # print('last_hidden[-1] size is:', last_hidden[-1].size())>>>[1, 16, 1024]
        context = self.attention(last_hidden[-1], encoder_outputs)
        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        # context = context.transpose(0, 1)  # (1,B,N)
        #print('context size is:', context.size())
        # Combine embedded input word and attended context, run through RNN
        #print('embedded size is:', embedded.size())
        context = context.transpose(0, 1)
        rnn_input = torch.cat([embedded, context], 2)
        #print('rnn_input size is:', rnn_input.size())
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        if(not trg is None):
            # print('trg size is:', trg.size())
            # print('trg max_len is:', trg.size(0))
            max_len = trg.size(0)
        else:
            max_len = src.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        print('encoder_output size is:', encoder_output.size())
        print('hidden size is:', hidden.size())

        hidden = hidden[:self.decoder.n_layers]
        #print('hidden hou size is:', hidden.size())
        output = torch.zeros(src.size(1)).long().cuda()  # sos
        #print('output original size is:', output.size())
        for t in range(1, max_len):
            #print('output size is:', output.size())
            output, hidden = self.decoder(
                    output, hidden, encoder_output)
            #print('decoder hidden size is:', hidden.size())
            outputs[t] = output
            # print('outputs[t] size is:', outputs[t].size())>>(16, 30)
            # print('output size is:', output.size())>>(16)

            if(not trg is None):
                is_teacher = random.random() < teacher_forcing_ratio
            else:
                is_teacher = False
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs

