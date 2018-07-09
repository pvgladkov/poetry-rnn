from __future__ import print_function

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM


class Vocabulary:
    def __init__(self, text, max_len):
        self.text = text
        self.max_len = max_len
        self.chars = sorted(list(set(text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))


class DataProvider:

    def __init__(self, text, max_len, logger):
        self.vocab = Vocabulary(text, max_len)
        self.logger = logger

    def get_data(self):
        max_len = self.vocab.max_len
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(self.vocab.text) - max_len, step):
            sentences.append(self.vocab.text[i: i + max_len])
            next_chars.append(self.vocab.text[i + max_len])

        X = np.zeros((len(sentences), max_len, len(self.vocab.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.vocab.chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, self.vocab.char_indices[char]] = 1
            y[i, self.vocab.char_indices[next_chars[i]]] = 1

        return X, y


class CharRNN:

    def __init__(self, vocabulary):
        self.vocab = vocabulary

    def build(self):

        model = Sequential()
        model.add(LSTM(128, input_shape=(self.vocab.max_len, len(self.vocab.chars))))
        model.add(Dense(len(self.vocab.chars)))
        model.add(Activation('softmax'))
        model.summary()

        return model

