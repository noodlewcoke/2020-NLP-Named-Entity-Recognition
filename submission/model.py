from typing import List
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np

class Model(torch.nn.Module):
    def __init__(self, 
                vocab_dim, 
                embedding_dim, 
                hidden_dim, 
                output_dim,
                char_vocab_dim, 
                char_embedding_dim,  
                char_hidden_dim,
                lr=1e-3,
                d_p=0.5):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.char_hidden_dim = char_hidden_dim

        self.char_embedding_layer = torch.nn.Embedding(char_vocab_dim, char_embedding_dim, padding_idx = 0)
        self.char_f_lstm = torch.nn.LSTM(char_embedding_dim, char_hidden_dim, batch_first=True, bidirectional=False)
        self.char_b_lstm = torch.nn.LSTM(char_embedding_dim, char_hidden_dim, batch_first=True, bidirectional=False)

        self.dropout = torch.nn.Dropout(d_p)
        self.embedding_layer = torch.nn.Embedding(vocab_dim, embedding_dim, padding_idx=0)
        self.bilstm = torch.nn.LSTM(embedding_dim + char_hidden_dim*2, hidden_dim, bidirectional=True)
  
        self.fc = torch.nn.Linear(2*hidden_dim + output_dim -1 , output_dim)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def character_embedding(self, sentences, max_len, device):
        embedded_sentence = []
        for sentence_char in sentences:
            lengths = np.array(list(map(len, sentence_char)))-1
            padded_tokens = pad_sequence(sentence_char, batch_first=True, padding_value=0)
            w = self.char_embedding_layer(padded_tokens.to(device))
            f = self.char_f_lstm(w)[0]
            f = f[np.arange(len(lengths)), lengths]
            b = self.char_b_lstm(w)[0][:, -1, :]
            w = torch.cat([f,b], dim = -1)
            n_words = len(w)
            pad = torch.zeros([max_len-n_words, w.size()[1]]).to(device)
            w = torch.cat([w, pad], dim=0)
            embedded_sentence.append(w)
        embedded_sentence = torch.stack(embedded_sentence, dim=0)

        return embedded_sentence

    def forward(self, sentences, characters, X_lengths):
        x = sentences
        x = self.embedding_layer(x)
        device = x.device
        chars = self.character_embedding(characters, int(torch.max(X_lengths).item()), device)
        
        x = torch.cat([x, chars], -1)
        x = self.dropout(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True)

        x, self.hidden = self.bilstm(x)        

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = x.transpose(0, 1)
        self.logits = []
        self.output = []

        prev_label = torch.zeros((x.size()[1], self.output_dim-1)).to(x.device)

        # 'Condition' each instance on the previous label
        for sentence in x:
            exp = torch.cat((sentence, prev_label), -1)
            exp = self.fc(exp)
            self.logits.append(exp)
            exp = F.softmax(exp[:,1:], dim=-1)
            self.output.append(exp)
            prev_label = exp
        self.logits = torch.stack(self.logits, 0).transpose(0,1)
        self.output = torch.stack(self.output, 0).transpose(0,1)

        return self.logits, self.output

    def update(self, sentence, characters, label, X_lengths):
        output, _ = self(sentence, characters, X_lengths)
        output = output.contiguous()
        output = output.view(-1, self.output_dim)

        label = label.contiguous()
        label = label.view(-1)

        self.optimizer.zero_grad()
        loss = self.loss_function(output, label)

        loss.backward()
        self.optimizer.step()
        return loss

    def evaluate(self, sentence, characters, label, X_lengths):
        with torch.no_grad():
            output, _ = self(sentence, characters, X_lengths)
            output = output.contiguous()

            output = output.view(-1, self.output_dim)

            label = label.contiguous()
            label = label.view(-1)

            loss = self.loss_function(output, label)
        return loss

    def prediction(self, sentence, characters, X_lengths):
        with torch.no_grad():
            _, output = self(sentence, characters, X_lengths)
            return output

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath, device):
        self.load_state_dict(torch.load(filepath, map_location=device))