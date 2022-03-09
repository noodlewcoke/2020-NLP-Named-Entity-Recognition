# 2020-NLP-Named-Entity-Recognition

2020

Implementation of Named Entity Recognition task for an NLP course as a homework.

The repo contains the submitted form of the homework in the _submission_ folder.

Named entity recognition (NER) is the task of identifying the named entities in a given text. The named entities can be names of people, locations, organizations, dates, quantities and so on. The objective is to classify this entities into pre-defined categories. In this case these categories are _ORG_ for organization, _PER_ for person, _LOC_ for location and _O_ for other. In this project we use a statistical approach using neural networks to classify each token in a given sentence with a NER tag.

The general approach is to use a bidirectional long short-term memory (BiLSTM) module to extract features from a given sentence and classify those using a softmax layer into categories. The BiLSTM takes word embeddings to avoid the sparsity problem that can occur from using one-hot encodings of the tokens. In this project nine different setups are explored in which four different models and three different preprocessing setups are used. The models are as described below:

- **BiLSTM** : This model uses a word embedding layer, a BiLSTM layer and a softmax layer. It is used as a baseline for the other models.
- **BiLSTM\_EXP** : For each instance, this experimental model conditions on the previous instance’s output. This model was inspired by the BiLSTM-CRF model, as that model conditions an instance to the outputs of the whole sequence. The idea is to treat the setup as having the Markovian property, assuming that the previous instance contains all the needed contextual information that is needed to classify current instance, and condition on this information.
- **BiLSTM\_CHAR** : This model adds character embeddings to the baseline model.
- **BiLSTM\_ULT** : This ’ultimate’ model combines all the extra features in the other models. Namely, character embeddings, Random UNK setup (see Section 2.1) and experimental part of BiLSTM EXP.
