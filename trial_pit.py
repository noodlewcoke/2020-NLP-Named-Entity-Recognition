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
from models import nerBiLSTM
from sklearn.metrics import precision_score as sk_precision
from tqdm import tqdm


def testing(exp_no):
    '''
    EXPERIMENT01
    '''
    exp_path = 'experiments/experiment' + exp_no + '/'
    # dev_losses = np.load(exp_path+'dev_losses.npy', allow_pickle=True)
    # test_accuracy = np.load(exp_path+'test_accuracy.npy', allow_pickle=True)
    # training_epoch_losses = np.load(exp_path+'training_epoch_losses.npy', allow_pickle=True)
    # training_losses = np.load(exp_path+'training_losses.npy', allow_pickle=True)
    test_path = 'data/testset.npy'
    with open('data/word_vocab.json', 'r') as f:
        word_vocab = json.load(f)
        f.close()
    with open('data/char_vocab.json', 'r') as f:
        char_vocab = json.load(f)
        f.close()

    vocab_dim = len(word_vocab['w2i'])
    testset = DataSetHandler(32, word_vocab, char_vocab)
    testset.load_enumerated_dataset(test_path, shuffle=False)
    model = nerBiLSTM(vocab_dim, 100, 50, 5)
    model.load(exp_path+'model')

    for w, c, y, l in testset:
        print(len(w[0]))
        break

if __name__ == '__main__':
    testing('01')