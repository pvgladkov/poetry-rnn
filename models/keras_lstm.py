from __future__ import print_function
from keras.callbacks import LambdaCallback, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys


class Model:

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

        def sample(preds, temperature=1.0):
            # helper function to sample an index from a probability array
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        def on_epoch_end(epoch, logs):
            # Function invoked at end of each epoch. Prints generated text.
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

        model.fit(x, y, batch_size=128, epochs=200, callbacks=[print_callback, tensor_board])
