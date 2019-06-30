import keras
import numpy as np

from nmt_utils import nmt_train_generator, bleu_score_enc_dec


def batch_generator(x, y, batch_size=64, shuffle=True, looping=True):
    indices = np.arange(x.shape[0])

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for idx in range(0, x.shape[0] - batch_size + 1, batch_size):
            batch_index = indices[idx:idx+batch_size]
            yield x[batch_index], y[batch_index]

        if not looping:
            raise StopIteration


class BleuLogger(keras.callbacks.Callback):

    def __init__(self, data, eval_every, batch_size, tar_vocab_size, encoder, decoder):
        self.src, self.tar = data
        self.generator = nmt_train_generator(self.src, self.tar,
                                             tar_vocab_size, batch_size)
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.encoder = encoder
        self.decoder = decoder

    def on_train_begin(self, logs={}):
        self.batch = 0
        self.losses = []
        self.scores = []

    def on_batch_end(self, batch, logs={}):
        if (batch + 1) % self.eval_every == 0:
            loss = np.mean([self.model.test_on_batch(*next(self.generator))
                            for _ in range(self.src.shape[0] // self.batch_size)])
            bleu = bleu_score_enc_dec(self.encoder, self.decoder, self.src, self.tar, self.batch_size)
            self.losses.append(loss)
            self.scores.append(bleu)