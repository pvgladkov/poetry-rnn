from __future__ import print_function

import sys
import random
import numpy as np

from keras.callbacks import LambdaCallback, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

from utils.functions import sample


class DataProvider:

    def __init__(self, text, max_len):
        print('corpus length:', len(text))
        self.text = text
        self.max_len = max_len
        self.chars = sorted(list(set(text)))
        print('total chars:', len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def get_data(self):
        max_len = self.max_len
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(self.text) - max_len, step):
            sentences.append(self.text[i: i + max_len])
            next_chars.append(self.text[i + max_len])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        x = np.zeros((len(sentences), max_len, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1

        return x, y


class CharRNN:

    def __init__(self, max_len, chars, char_indices, indices_char, text):
        self.max_len = max_len
        self.chars = chars
        self.char_indices = char_indices
        self.indices_char = indices_char
        self.text = text

    def fit(self, x, y):

        model = Sequential()
        model.add(LSTM(128, input_shape=(self.max_len, len(self.chars))))
        model.add(Dense(len(self.chars)))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        def on_epoch_end(epoch):
            print()
            print('----- Generating text after Epoch: %d' % epoch)

            start_index = random.randint(0, len(self.text) - self.max_len - 1)
            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print('----- diversity:', diversity)

                generated = ''
                sentence = self.text[start_index: start_index + self.max_len]
                generated += sentence
                print('----- Generating with seed: "' + sentence + '"')
                sys.stdout.write(generated)

                for i in range(400):
                    x_pred = np.zeros((1, self.max_len, len(self.chars)))
                    for t, char in enumerate(sentence):
                        x_pred[0, t, self.char_indices[char]] = 1.

                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = sample(preds, diversity)
                    next_char = self.indices_char[next_index]

                    generated += next_char
                    sentence = sentence[1:] + next_char

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                print()

        print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
        tensor_board = TensorBoard(histogram_freq=0, write_grads=True, write_images=True)

        model.fit(x, y, batch_size=128, epochs=400, callbacks=[print_callback, tensor_board])
