import numpy as np
from typing import List, Tuple
import json
from model import Model
import torch


def build_model(device: str) -> Model:
    return StudentModel(device)


class RandomBaseline(Model):

    options = [
        ('LOC', 98412),
        ('O', 2512990),
        ('ORG', 71633),
        ('PER', 115758)
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x] for x in tokens]


class StudentModel(Model):

    def __init__(self, device):
        Model.__init__(self, vocab_dim=138554, embedding_dim=100, hidden_dim=50, output_dim=5,
                        char_vocab_dim=771, char_embedding_dim=50, char_hidden_dim=25, d_p=0.0) # vocab_dim, embedding_dim, hidden_dim, output_dim, lr=1e-3
        self.device = device
        with open('model/word_vocab.json', 'r') as f:
            self.word_vocab = json.load(f)
            f.close()
        with open('model/char_vocab.json', 'r') as f:
            self.char_vocab = json.load(f)
            f.close()

        self.load('model/model', self.device)
    
    def predict(self, tokens: List[List[str]]) -> List[List[str]]:

        predictions = []
        tags = ['ORG', 'PER', 'LOC', 'O']
        for sentence in tokens:
            enum_sentence = []
            enum_characters = []
            for token in sentence:
                try:
                    enum_sentence.append(self.word_vocab['w2i'][token])
                except KeyError:
                    enum_sentence.append(self.word_vocab['w2i']['<UNK>'])
                except:
                    print("Something went wrong with the word dictionary.")
                enum_word = []
                for c in token:
                    try:
                        enum_word.append(self.char_vocab['c2i'][c])
                    except KeyError:
                        enum_word.append(self.char_vocab['c2i']['<UNK>'])
                    except:
                        print("Something went wrong with the char dictionary.")
                enum_characters.append(torch.LongTensor(enum_word))
            sentence_length = torch.LongTensor([len(enum_sentence)]).to(self.device)
            enum_sentence = torch.LongTensor(enum_sentence).view(1, -1).to(self.device)
            prediction = self.prediction(enum_sentence, [enum_characters], sentence_length).cpu().numpy()
            try:
                prediction = [tags[i] for i in np.argmax(prediction, axis = -1).squeeze()]
            except TypeError:
                prediction = [tags[np.argmax(prediction, axis = -1).squeeze()]]
            except:
                raise Exception([tags[np.argmax(prediction, axis = -1).squeeze()]])

            predictions.append(prediction)
        return predictions



        
