import io
import numpy as np
from models.char_rnn import CharRNN, DataProvider

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

    max_len = 40
    data_provider = DataProvider(text, max_len=40)

    x, y = data_provider.get_data()
    chars = data_provider.chars
    char_indices = data_provider.char_indices
    indices_char = data_provider.indices_char

    model = CharRNN(max_len, chars, char_indices, indices_char, text)
    model.fit(x, y)