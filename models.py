import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import csv
import json
import re
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class nerBiLSTM(torch.nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_dim, output_dim, lr=1e-3):
        super(nerBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding_layer = torch.nn.Embedding(vocab_dim, embedding_dim, padding_idx=0)
        self.bilstm = torch.nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        self.fc = torch.nn.Linear(2*hidden_dim, output_dim)
        # self.hidden = self.init_hidden()

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, sentence, X_lengths):
        x = sentence 
        x = self.embedding_layer(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True)

        x, self.hidden = self.bilstm(x)        

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        self.logits = self.fc(x)
        self.output = F.softmax(self.logits, dim=2)
        return self.logits, self.output

    def update(self, sentence, label, X_lengths):
        output, _ = self(sentence, X_lengths)

        output = output.view(-1, self.output_dim)

        label = label.contiguous()
        label = label.view(-1)

        self.optimizer.zero_grad()
        loss = self.loss_function(output, label)

        loss.backward()
        self.optimizer.step()
        return loss

    # def init_hidden(self):
    #     # Before we've done anything, we dont have any hidden state.
    #     # Refer to the Pytorch documentation to see exactly
    #     # why they have this dimensionality.
    #     # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
    #     return (Variable(torch.zeros(2, 5, self.hidden_dim)),   
    #             Variable(torch.zeros(2, 5, self.hidden_dim)))
    def evaluate(self, sentence, label, X_lengths):
        with torch.no_grad():
            output, _ = self(sentence, X_lengths)

            output = output.view(-1, self.output_dim)

            label = label.contiguous()
            label = label.view(-1)

            loss = self.loss_function(output, label)
        return loss

    def prediction(self, sentence, X_lengths):
        with torch.no_grad():
            _, output = self(sentence, X_lengths)
            return output

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location='cpu'))



class nerBiLSTM_char(torch.nn.Module):
    def __init__(self, 
                vocab_dim, 
                embedding_dim, 
                hidden_dim, 
                output_dim,
                char_vocab_dim, 
                char_embedding_dim,  
                char_hidden_dim,
                lr=1e-3):
        super(nerBiLSTM_char, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.char_hidden_dim = char_hidden_dim

        self.char_embedding_layer = torch.nn.Embedding(char_vocab_dim, char_embedding_dim, padding_idx = 0)
        self.char_f_lstm = torch.nn.LSTM(char_embedding_dim, char_hidden_dim, batch_first=True, bidirectional=False)
        self.char_b_lstm = torch.nn.LSTM(char_embedding_dim, char_hidden_dim, batch_first=True, bidirectional=False)


        self.embedding_layer = torch.nn.Embedding(vocab_dim, embedding_dim, padding_idx=0)
        self.bilstm = torch.nn.LSTM(embedding_dim + char_hidden_dim*2, hidden_dim, bidirectional=True)

        self.fc = torch.nn.Linear(2*hidden_dim, output_dim)
        # self.hidden = self.init_hidden()

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def character_embedding(self, sentences, max_len):
        embedded_sentence = []
        for sentence_char in sentences:
            # embedded_words = []
            # for word in sentence_char:
            #     w = word.unsqueeze(0)
            #     w = self.char_embedding_layer(w)
            #     f = self.char_f_lstm(w)[0][0, -1, :]
            #     b = torch.flip(w, [0,1])
            #     b = self.char_b_lstm(b)[0][0, -1, :]
            #     w = torch.cat([f,b], dim=-1)

            #     embedded_words.append(w)

            lengths = np.array(list(map(len, sentence_char)))-1
            padded_tokens = pad_sequence(sentence_char, batch_first=True, padding_value=0)
            w = self.char_embedding_layer(padded_tokens)
            f = self.char_f_lstm(w)[0]
            f = f[np.arange(len(lengths)), lengths]
            b = self.char_b_lstm(w)[0][:, -1, :]
            w = torch.cat([f,b], dim = -1)
            n_words = len(w)
            pad = torch.zeros([max_len-n_words, w.size()[1]])
            w = torch.cat([w, pad], dim=0)
            # for i in range(n_words, max_len):
                # embedded_words.append(torch.zeros_like(embedded_words[0]))
            embedded_sentence.append(w)

        # print(pad_sequence(embedded_sentence, batch_first=True, padding_value=torch.zeros_like(embedded_sentence[0][0])).size())
            # embedded_sentence.append(torch.stack(embedded_words, dim=0))
        embedded_sentence = torch.stack(embedded_sentence, dim=0)

        return embedded_sentence

    def forward(self, sentences, characters, X_lengths):
        x = sentences
        x = self.embedding_layer(x)

        chars = self.character_embedding(characters, torch.max(X_lengths))

        x = torch.cat([x, chars], -1)
        print(x.size())
        x = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True)

        x, self.hidden = self.bilstm(x)        

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        self.logits = self.fc(x)
        self.output = F.softmax(self.logits, dim=2)
        return self.logits, self.output

    def update(self, sentence, label, X_lengths):
        output, _ = self(sentence, X_lengths)

        output = output.view(-1, self.output_dim)

        label = label.contiguous()
        label = label.view(-1)

        self.optimizer.zero_grad()
        loss = self.loss_function(output, label)

        loss.backward()
        self.optimizer.step()
        return loss

    # def init_hidden(self):
    #     # Before we've done anything, we dont have any hidden state.
    #     # Refer to the Pytorch documentation to see exactly
    #     # why they have this dimensionality.
    #     # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
    #     return (Variable(torch.zeros(2, 5, self.hidden_dim)),   
    #             Variable(torch.zeros(2, 5, self.hidden_dim)))
    def evaluate(self, sentence, label, X_lengths):
        with torch.no_grad():
            output, _ = self(sentence, X_lengths)

            output = output.view(-1, self.output_dim)

            label = label.contiguous()
            label = label.view(-1)

            loss = self.loss_function(output, label)
        return loss

    def prediction(self, sentence, X_lengths):
        with torch.no_grad():
            _, output = self(sentence, X_lengths)
            return output

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location='cpu'))

