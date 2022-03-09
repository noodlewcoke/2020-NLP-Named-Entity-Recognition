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
from data import DataSetHandler
from models import nerBiLSTM, nerBiLSTM_char
from sklearn.metrics import precision_score as sk_precision
from tqdm import tqdm

def main():
    train_path = 'data/trainset_truecase_enumerated.npy'
    dev_path = 'data/devset.npy'
    test_path = 'data/testset.npy'
    with open('data/word_vocab.json', 'r') as f:
        word_vocab = json.load(f)
        f.close()
    with open('data/char_vocab.json', 'r') as f:
        char_vocab = json.load(f)
        f.close()

    vocab_dim = len(word_vocab['w2i'])
    char_dim = len(char_vocab['c2i'])
    trainset = DataSetHandler(32, word_vocab, char_vocab)
    trainset.load_enumerated_dataset(train_path, shuffle=True)
    model = nerBiLSTM_char(vocab_dim, 100, 50, 5, char_dim, 50, 25)
    
    for w, c, y, l in trainset:
        print(w.size())
        print(c[0])
        model(w, c, torch.LongTensor(l))

        break

if __name__ == '__main__':
    main()