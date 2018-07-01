import io
from models import char_rnn, word_rnn
import argparse


def get_text():
    txt = ''
    with io.open('data/blok_1.txt', encoding='utf-8') as f:
        txt += f.read().lower()
    with io.open('data/blok_2.txt', encoding='utf-8') as f:
        txt += f.read().lower()
    with io.open('data/blok_3.txt', encoding='utf-8') as f:
        txt += f.read().lower()
    with io.open('data/blok_4.txt', encoding='utf-8') as f:
        txt += f.read().lower()
    with io.open('data/blok_5.txt', encoding='utf-8') as f:
        txt += f.read().lower()
    with io.open('data/blok_6.txt', encoding='utf-8') as f:
        txt += f.read().lower()
    with io.open('data/blok_7.txt', encoding='utf-8') as f:
        txt += f.read().lower()
    return txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='char', help='char or word', choices=['char', 'word'])
    args = parser.parse_args()

    text = get_text()

    if args.type == 'char':
        max_len = 40
        data_provider = char_rnn.DataProvider(text, max_len)

        X, y = data_provider.get_data()
        chars = data_provider.chars
        char_indices = data_provider.char_indices
        indices_char = data_provider.indices_char

        model = char_rnn.CharRNN(max_len, chars, char_indices, indices_char, text)
        model.fit(X, y)

    elif args.type == 'word':
        max_len = 20
        data_provider = word_rnn.DataProvider(text, max_len)

        X, y = data_provider.get_data()

        model = word_rnn.WordRNN(max_len,
                                 data_provider.words,
                                 data_provider.list_words,
                                 data_provider.word_indices,
                                 data_provider.indices_word,
                                 text,
                                 data_provider.embedding_dim,
                                 data_provider.embedding_matrix())
        model.fit(X, y)