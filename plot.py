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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn 

n_experiment = '10'
n_model = 'BiLSTM_mixedcase'

exp_dir = 'experiments/experiment' + n_experiment + '/'

dev_losses = np.load(exp_dir+'dev_losses.npy', allow_pickle=True)
train_epoch_losses = np.load(exp_dir+'training_epoch_losses.npy', allow_pickle=True)
train_losses = np.load(exp_dir+'training_losses.npy', allow_pickle=True)
test_accuracy = np.load(exp_dir+'test_accuracy.npy', allow_pickle=True)


# extra_dev_losses = np.load(exp_dir+'extra_losses/dev_losses.npy', allow_pickle=True)
# extra_train_epoch_losses = np.load(exp_dir+'extra_losses/training_epoch_losses.npy', allow_pickle=True)
# extra_train_losses = np.load(exp_dir+'extra_losses/training_losses.npy', allow_pickle=True)

# dev_losses = np.concatenate([dev_losses, extra_dev_losses], axis=-1)
# train_epoch_losses = np.concatenate([train_epoch_losses, extra_train_epoch_losses], axis=-1)
# print(len(dev_losses))
# print(len(extra_dev_losses))
# exit()
plt.subplot(121)
plt.title('Training Losses')
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epochs')
plt.plot(np.arange(len(train_epoch_losses)), train_epoch_losses)
# plt.show()
plt.subplot(122)
plt.title('Validation Losses')
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epochs')
plt.plot(np.arange(len(dev_losses)), dev_losses)
# plt.savefig('plots/'+n_model)
plt.show()

# plt.plot(np.arange(len(test_accuracy)), test_accuracy)
# plt.show()