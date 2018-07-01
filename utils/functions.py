import numpy as np
import sys
import random


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def char_generate(model, max_len, chars, sentence, char_indices, indices_char, diversity):
    x_pred = np.zeros((1, max_len, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]
    return next_char


def sentences_to_indices(sentences, word_to_index, max_len):
    X = np.zeros((len(sentences), max_len))
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for j, word in enumerate(words):
            X[i, j] = word_to_index.get(word)
    return X


def word_generate(model, max_len, sentence, word_indices, indices_word, diversity):
    x_pred = sentences_to_indices([sentence], word_indices, max_len)
    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, diversity)
    return indices_word[next_index]


def on_epoch_end_word(epoch, logs, model, vocab, logger):

    logger.info('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(vocab.text) - vocab.max_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        logger.info('----- diversity: {}'.format(diversity))

        generated = ''
        sentence = ' '.join(vocab.list_words[start_index: start_index + vocab.max_len])
        generated += sentence
        logger.info('----- Generating with seed: "' + sentence + '"')

        for i in range(400):
            next_word = word_generate(model, vocab.max_len, sentence,
                                      vocab.word_indices, vocab.indices_word, diversity)

            generated = generated + ' ' + next_word
            sentence = ' '.join(sentence.split()[1:] + [next_word])

        logger.info(generated)


def on_epoch_end_char(epoch, logs, model, vocab, logger):
    logger.info('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(vocab.text) - vocab.max_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        logger.info('----- diversity: {}'.format(diversity))

        generated = ''
        sentence = vocab.text[start_index: start_index + vocab.max_len]
        generated += sentence
        logger.info('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            next_char = char_generate(model, vocab.max_len, vocab.chars, sentence,
                                      vocab.char_indices, vocab.indices_char, diversity)

            generated += next_char
            sentence = sentence[1:] + next_char

        logger.info(generated)