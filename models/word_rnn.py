import sys
import random

import numpy as np
from gensim.models import KeyedVectors
from keras.layers import Input, LSTM, Dropout, Dense, Activation, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, TensorBoard

from utils.functions import sentences_to_indices, word_generate


class DataProvider:

    def __init__(self, text, max_len):
        self.text = text
        self.word2vec = KeyedVectors.load_word2vec_format('./embeddings/word2vec')
        self.max_len = max_len
        self.list_words = self.text.lower().split()
        self.words = set(self.list_words)

        self.word_indices = dict((w, i) for i, w in enumerate(self.words))
        self.indices_word = dict((i, w) for i, w in enumerate(self.words))

        self.embedding_dim = self.word2vec.vector_size

    def get_data(self):
        sentences = []
        next_words = []
        step = 3

        max_len = self.max_len

        for i in range(0, len(self.list_words) - max_len, step):
            sentence = ' '.join(self.list_words[i: i + max_len])
            sentences.append(sentence)
            next_words.append(self.list_words[i + max_len])

        X = sentences_to_indices(sentences, self.word_indices, self.max_len)
        y = np.zeros((len(sentences), len(self.words)))
        for i, w in enumerate(next_words):
            word_index = self.word_indices.get(w)
            y[i, word_index] = 1

        return X, y

    def embedding_matrix(self):
        X = np.zeros((len(self.word_indices), self.embedding_dim))
        for word, i in self.word_indices.items():
            if word in self.word2vec:
                X[i, :] = self.word2vec[word]
            else:
                X[i, :] = np.zeros((self.embedding_dim, ))
        return X


class WordRNN:

    def __init__(self, max_len, words, list_words, word_indices, indices_word, text, embedding_dim, emb_matrix):
        self.max_len = max_len
        self.words = words
        self.list_words = list_words
        self.emb_matrix = emb_matrix
        self.embedding_dim = embedding_dim
        self.text = text
        self.indices_word = indices_word
        self.word_indices = word_indices

    def fit(self, x, y):

        sentence_indices = Input(shape=(self.max_len,), dtype=np.int32)

        embedding = Embedding(input_dim=len(self.words), output_dim=self.embedding_dim)
        embedding.build((None, ))
        embedding.set_weights([self.emb_matrix])

        X = embedding(sentence_indices)
        X = LSTM(units=128, return_sequences=True)(X)
        X = Dropout(0.5)(X)
        X = LSTM(units=128, return_sequences=False)(X)
        X = Dropout(0.5)(X)
        X = Dense(len(self.words), activation='softmax')(X)
        X = Activation('softmax')(X)
        model = Model(inputs=sentence_indices, outputs=X)

        model.summary()

        optimizer = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        def on_epoch_end(epoch, logs):
            print()
            print('----- Generating text after Epoch: %d' % epoch)

            start_index = random.randint(0, len(self.text) - self.max_len - 1)
            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print('----- diversity:', diversity)

                generated = ''
                sentence = ' '.join(self.list_words[start_index: start_index + self.max_len])
                generated += sentence
                print('----- Generating with seed: "' + sentence + '"')
                sys.stdout.write(generated)

                for i in range(400):
                    next_word = word_generate(model, self.max_len, sentence,
                                              self.word_indices, self.indices_word, diversity)

                    generated = generated + ' ' + next_word
                    sentence = ' '.join(sentence.split()[1:] + [next_word])

                    sys.stdout.write(next_word)
                    sys.stdout.flush()
                print()

        print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
        tensor_board = TensorBoard(histogram_freq=0, write_grads=True, write_images=True)

        model.fit(x, y, batch_size=128, epochs=200, callbacks=[print_callback, tensor_board])