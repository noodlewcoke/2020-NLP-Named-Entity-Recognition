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
from tqdm import tqdm
from multiprocessing import Pool

def vocabularize(paths, save_path=None):
    '''
    # This function creates a vocabulary containing both cased and lowercased words
    # The purpose of this is to train NER model with mixture of cased and lowercased sentences 
    # which was proved to work best on average for cased and lowercased datasets
    '''
    #! Maybe remove empty space from the char_vocab
    characters = ''
    words = []

    #* Word vocabulary    
    for path in paths:
        with open(path, 'r') as f:
            for sentence in f:
                sentence = sentence.strip()
                if sentence.startswith('#'):
                    lower_sentence = sentence.lower()
                    lower_sentence = lower_sentence.split()
                    sentence = sentence.split()
                    unique_words = list(set(sentence[1:]+lower_sentence[1:]))
                    words.extend(unique_words)
            f.close()

    words = sorted(list(set(words)))
    words.insert(0, '<UNK>')
    words.insert(0, '<PAD>')
    wordtoindex = {v: k for k, v in enumerate(words)}
    indextoword = {k: v for k, v in enumerate(words)}
    vocab = {'w2i' : wordtoindex,
             'i2w' : indextoword}

    #* Char vocabulary
    for path in paths:
            with open(path, 'r') as f:
                reader = f.read()
                reader += reader.lower()
                characters += reader
                f.close()
    characters = sorted(list(set(characters)))
    characters.remove('\n')
    characters.remove('\t')
    characters.insert(0, '<UNK>')
    characters.insert(0, '<PAD>')

    chartoindex = {v: k for k, v in enumerate(characters)}
    indextochar = {k: v for k, v in enumerate(characters)}
    char_vocab = {  'c2i' : chartoindex,
                    'i2c' : indextochar}

    print("Word vocabulary size: ", len(words))
    print("Char vocabulary size: ", len(characters))
    if save_path:
        np.save(save_path+'wordsList', words)
        with open(save_path+'word_vocab.json', 'w') as f:
            json.dump(vocab, f)
            f.close()
        np.save(save_path+'charList', characters)
        with open(save_path+'char_vocab.json', 'w') as f:
            json.dump(char_vocab, f)
            f.close()
    return char_vocab, vocab, characters, words


class DataSetHandler:
    '''
    A class for building a dataset out of provided data, given also vocabularies for words and characters.
    '''
    def __init__(self, 
                batch_size, 
                word_vocabulary, 
                char_vocabulary,
                ignore_singletons = False):
        self.batch_size = batch_size
        self.word_vocabulary = word_vocabulary['w2i']
        self.char_vocabulary = char_vocabulary['c2i']        
        self.enumerated = False
        self.str_dataset = False
        self.counter = 0
        self.ignore_singletons = ignore_singletons

    def load_dataset(self, file_path, shuffle=False):
        self.dataset = np.load(file_path, allow_pickle=True)
        self.data_len = len(self.dataset)
        self.str_dataset = True
        if shuffle: np.random.shuffle(self.dataset)
        print('Dataset in {} loaded. Number of samples: {}'.format(file_path, self.data_len))

    def load_enumerated_dataset(self, file_path, shuffle=False):
        self.enum_dataset = np.load(file_path, allow_pickle=True)
        self.data_len = len(self.enum_dataset)
        self.enumerated = True
        if shuffle: np.random.shuffle(self.enum_dataset)
        print('Enumerated dataset in {} loaded. Number of samples: {}'.format(file_path, self.data_len))


    def create_dataset(self, file_path, mode = 'true', shuffle=False):
        self.mode = mode.lower()
        assert self.mode == 'true' or self.mode == 'lower' or self.mode == 'mixed', "Mode variable, denoting casing, should either be 'true', 'lower' or 'mixed'."

        with open(file_path, 'r') as tsv:
            sentences = tsv.read().split('#')[1:]
            tsv.close()

        self.dataset = []
        for data in sentences:
            data = data.strip().split('\n')
            assert len(data[0].split()) == len(data[1:])

            tokens, chars, labels = [], [], []
            lower_tokens, lower_chars = [], []
            for line in data[1:]:
                _, token, label = line.split('\t')

                #* Case mode handling
                if self.mode == 'lower':
                    token = token.lower()
                elif self.mode == 'mixed':
                    lower_token = token.lower()
                    lower_tokens.append(lower_token)
                    lower_chars.append(list(lower_token))

                tokens.append(token)
                chars.append(list(token))

                #* Label handling : ORG -> 1, PER -> 2, LOC -> 3, O -> 4
                if label == 'ORG':
                    label = 1
                elif label == 'PER':
                    label = 2
                elif label == 'LOC':
                    label = 3
                elif label == 'O':
                    label = 4
                else:
                    print("Problem with this label: ", label)
                labels.append(label)

            self.dataset.append((tokens, chars, labels))
            if self.mode == 'mixed':
                self.dataset.append((lower_tokens, lower_chars, labels))
        del sentences
        print('Dataset has been created.')

        self.data_len = len(self.dataset)
        self.str_dataset = True
        if shuffle: np.random.shuffle(self.dataset)
        if self.ignore_singletons: self.tokenfreq()

    def enumerate(self, data):
        enum_data = []
        for sentence, chars, labels in tqdm(data):
            enum_sentence = []
            enum_chars = []
            for word, cword in zip(sentence, chars):
                if word in self.wordkeys:
                    if self.ignore_singletons and word in self.least_frequent and np.random.rand(1)[0] < 0.5:
                        '''
                        This replaces tokens with appearance less than one, with the <UNK> tag, with probability 0.5 . 
                        '''
                        enum_sentence.append(self.word_vocabulary['<UNK>'])
                    else:
                        enum_sentence.append(self.word_vocabulary[word])
                else:
                    enum_sentence.append(self.word_vocabulary['<UNK>'])
                enum_char = []
                for char in cword:
                    if char in self.charkeys:
                        enum_char.append(self.char_vocabulary[char])
                    else:
                        enum_char.append(self.char_vocabulary['<UNK>'])
                enum_chars.append(enum_char)
            enum_data.append((enum_sentence, enum_chars, labels))
        return enum_data


    def token2index(self, num_workers=1):
        print('Enumerating the dataset.')
        
        self.enum_dataset = []
        self.wordkeys = list(self.word_vocabulary.keys())
        self.charkeys = list(self.char_vocabulary.keys())
        lenght = len(self.dataset)
        data1 = self.dataset[:lenght//4]
        data2 = self.dataset[lenght//4 : lenght//2]
        data3 = self.dataset[lenght//2 : 3*lenght//4]
        data4 = self.dataset[3*lenght//4 :]
        results = []
        with Pool(4) as p:
            results = p.map(self.enumerate, [data1, data2, data3, data4])
        self.enum_dataset = [sample for data in results for sample in data]
    
        assert len(self.dataset) == len(self.enum_dataset), "Mistakes were made. Lenghts of the datasets don't match."
        self.enumerated = True
        print("Dataset has been enumerated.")
        return self.enum_dataset

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.data_len and self.enumerated:
            last = min(self.counter+self.batch_size, self.data_len)
            batch = self.enum_dataset[self.counter:last]
            batch = np.array(sorted(batch, key=lambda d: len(d[0]), reverse=True))
            data_w = batch[:, 0]
            data_c = batch[:, 1]
            data_y = batch[:, 2]
            sentence_lengths = [len(i) for i in data_w]

            #* Tokens and char to indices
            data_w = [torch.LongTensor(sentence) for sentence in data_w] 
            data_c = [[torch.LongTensor(word) for word in sentence] for sentence in data_c]

            y = [torch.LongTensor(i) for i in data_y]
            padded_w = pad_sequence(data_w, batch_first=True, padding_value=self.word_vocabulary['<PAD>'])

            # Check the log file *05_.
            # padded_c = []
            # for sentence in data_c:
            #     padded_c.append(pad_sequence(sentence, batch_first=True, padding_value=self.char_vocabulary['<PAD>']))
            # padded_c = pad_sequence(data_c, batch_first=True, padding_value=self.char_vocabulary['<PAD>'])

            padded_y = pad_sequence(y, batch_first=True, padding_value=0)
            self.counter = last
            return padded_w, data_c, padded_y, sentence_lengths
        else:
            if not self.enumerated:
                raise ValueError('Dataset must be enumerated first. Use the token2index() funtion.')
            self.counter = 0
            raise StopIteration()

    def save(self, save_path):
        if self.str_dataset:
            np.save(save_path, self.dataset, allow_pickle=True)
            print('Dataset has been saved in {}'.format(save_path))
        if self.enumerated:
            np.save(save_path+'_enumerated', self.enum_dataset, allow_pickle=True)
            print('Enumerated dataset has been saved in {}'.format(save_path+'_enumerated'))
        if not self.str_dataset and not self.enumerated:
            raise ValueError('There are no datasets to save.')

    def tokenfreq(self):
        '''
        Use this function only with the training set!
        Compute the frequency of each token.
        This is utilized later to replace least frequent tokens with the <UNK> tag.
        '''

        assert self.str_dataset, 'Dataset should be in string format.'

        self.token_fq = {k: 0 for k in self.word_vocabulary.keys()}
        for sentence, _, _ in self.dataset:
            for token in sentence:
                try:
                    self.token_fq[token] += 1
                except:
                    print("Unexpected token: ", token)

        aslist = sorted(list(self.token_fq.items()), key = lambda x: x[1])
        self.least_frequent = []
        n_replaced = 0
        for t, f in aslist:
            if f < 2:
                self.least_frequent.append(t)
        self.ignore_singletons = True

        return self.token_fq, self.least_frequent

def create_vocabulary():
    trainFileName = 'data/' + 'train.tsv'
    # devFileName = 'data/' + 'dev.tsv'

    char_vocab, vocab, characters, words = vocabularize([trainFileName], 'data/')
    print(len(words))
    print(vocab['i2w'][0])
    print(list(vocab['w2i'].keys())[:10])
    print(len(characters))
    print(characters[:10])
    print(list(char_vocab['c2i'].keys())[:10])

def main():
    # create_vocabulary()

    trainFileName = 'data/' + 'train.tsv'
    devFileName = 'data/' + 'dev.tsv'
    testFileName = 'data/' + 'test.tsv'
    with open('data/word_vocab.json', 'r') as f:
        word_vocab = json.load(f)
        f.close()
    with open('data/char_vocab.json', 'r') as f:
        char_vocab = json.load(f)
        f.close()
    
    dataset = DataSetHandler(32, word_vocab, char_vocab)
    dataset.create_dataset(trainFileName, mode='true', shuffle=True)
    dataset.token2index()
    dataset.save('data/trainset_truecase')
    # dataset.load_dataset('data/trainset.npy')
    print(dataset.data_len)


if __name__ == '__main__':
    main()