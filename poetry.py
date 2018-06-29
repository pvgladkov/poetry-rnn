import io
import numpy as np
from models.keras_lstm import Model

if __name__ == '__main__':
    text = ''
    with io.open('data/blok_1.txt', encoding='utf-8') as f:
        text += f.read().lower()
    with io.open('data/blok_2.txt', encoding='utf-8') as f:
        text += f.read().lower()
    with io.open('data/blok_3.txt', encoding='utf-8') as f:
        text += f.read().lower()
    with io.open('data/blok_4.txt', encoding='utf-8') as f:
        text += f.read().lower()
    with io.open('data/blok_5.txt', encoding='utf-8') as f:
        text += f.read().lower()
    with io.open('data/blok_6.txt', encoding='utf-8') as f:
        text += f.read().lower()
    with io.open('data/blok_7.txt', encoding='utf-8') as f:
        text += f.read().lower()

    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    max_len = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - max_len, step):
        sentences.append(text[i: i + max_len])
        next_chars.append(text[i + max_len])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    model = Model(max_len, chars, char_indices, indices_char, text)
    model.fit(x, y)