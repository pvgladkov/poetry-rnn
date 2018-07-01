import numpy as np
from gensim.models import KeyedVectors
from keras.layers import Input, LSTM, Dropout, Dense, Activation, Embedding
from keras.models import Model


from utils.functions import sentences_to_indices


class Vocabulary:

    def __init__(self, text, max_len):
        self.text = text
        self.max_len = max_len

        self.list_words = self.text.lower().split()
        self.words = set(self.list_words)

        self.word_indices = dict((w, i) for i, w in enumerate(self.words))
        self.indices_word = dict((i, w) for i, w in enumerate(self.words))


class DataProvider:

    def __init__(self, text, max_len):
        self.vocab = Vocabulary(text, max_len)
        self.word2vec = KeyedVectors.load_word2vec_format('./embeddings/word2vec')
        self.embedding_dim = self.word2vec.vector_size

    def get_data(self):
        sentences = []
        next_words = []
        step = 3

        max_len = self.vocab.max_len

        for i in range(0, len(self.vocab.list_words) - max_len, step):
            sentence = ' '.join(self.vocab.list_words[i: i + max_len])
            sentences.append(sentence)
            next_words.append(self.vocab.list_words[i + max_len])

        X = sentences_to_indices(sentences, self.vocab.word_indices, max_len)
        y = np.zeros((len(sentences), len(self.vocab.words)))
        for i, w in enumerate(next_words):
            word_index = self.vocab.word_indices.get(w)
            y[i, word_index] = 1

        return X, y

    def embedding_matrix(self):
        X = np.zeros((len(self.vocab.word_indices), self.embedding_dim))
        for word, i in self.vocab.word_indices.items():
            if word in self.word2vec:
                X[i, :] = self.word2vec[word]
            else:
                X[i, :] = np.zeros((self.embedding_dim, ))
        return X


class WordRNN:

    def __init__(self, vocabulary, embedding_dim, emb_matrix):
        self.vocab = vocabulary
        self.emb_matrix = emb_matrix
        self.embedding_dim = embedding_dim

    def build(self):

        sentence_indices = Input(shape=(self.vocab.max_len, ), dtype=np.int32)

        embedding = Embedding(input_dim=len(self.vocab.words), output_dim=self.embedding_dim)
        embedding.build((None, ))
        embedding.set_weights([self.emb_matrix])

        X = embedding(sentence_indices)
        X = LSTM(units=128, return_sequences=True)(X)
        X = Dropout(0.5)(X)
        X = LSTM(units=128, return_sequences=False)(X)
        X = Dropout(0.5)(X)
        X = Dense(len(self.vocab.words), activation='softmax')(X)
        X = Activation('softmax')(X)
        model = Model(inputs=sentence_indices, outputs=X)

        model.summary()

        return model