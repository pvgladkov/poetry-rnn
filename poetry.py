import io
from models import char_rnn, word_rnn
import argparse

from keras.optimizers import Adam, RMSprop
from keras.callbacks import LambdaCallback, TensorBoard
from utils.functions import on_epoch_end_word, on_epoch_end_char

import logging


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

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='char', help='char or word', choices=['char', 'word'])
    args = parser.parse_args()

    text = get_text()

    if args.type == 'char':
        max_len = 40
        data_provider = char_rnn.DataProvider(text, max_len, logger)

        X, y = data_provider.get_data()

        model = char_rnn.CharRNN(data_provider.vocab)
        model = model.build()

        optimizer = RMSprop(lr=0.01)

        def epoch_callback(epoch, logs):
            return on_epoch_end_char(epoch, logs, model, data_provider.vocab, logger)

    else:
        max_len = 10
        data_provider = word_rnn.DataProvider(text, max_len, logger)

        X, y = data_provider.get_data()

        model = word_rnn.WordRNN(data_provider.vocab,
                                 data_provider.embedding_dim,
                                 data_provider.embedding_matrix())

        model = model.build()
        optimizer = Adam()

        def epoch_callback(epoch, logs):
            return on_epoch_end_word(epoch, logs, model, data_provider.vocab, logger)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print_callback = LambdaCallback(on_epoch_end=epoch_callback)
    tensor_board = TensorBoard(histogram_freq=0, write_grads=True, write_images=True)

    model.fit(X, y, batch_size=128, epochs=200, callbacks=[print_callback, tensor_board])
