import keras
import numpy as np
import pandas as pd

from nmt_utils import nmt_train_generator, bleu_score_enc_dec
from wmt import load_wmt
from model import define_nmt


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


if __name__ == '__main__':
    np.random.seed(42)
    en_train, en_test, en_tokenizer, de_train, de_test, de_tokenizer = load_wmt(split=0.3)
    en_vocab_size, de_vocab_size = len(en_tokenizer), len(de_tokenizer)

    batch_size = 64
    val_size = 3200
    en_train_t, en_train_v = en_train[val_size:], en_train[:val_size]
    de_train_t, de_train_v = de_train[val_size:], de_train[:val_size]

    # model parameters
    hidden_size = 96
    embedding_size = 100
    timesteps = 30

    # hyperparameters
    lr = 0.001
    dropout = 0.2

    model, encoder_model, decoder_model = define_nmt(
        hidden_size, embedding_size,
        timesteps, en_vocab_size, de_vocab_size, dropout, lr)

    train_generator = nmt_train_generator(en_train_t, de_train_t, de_vocab_size, batch_size)

    steps = en_train.shape[0]//batch_size
    eval_every = 100

    bleu_logger = BleuLogger((en_train_v, de_train_v), eval_every, batch_size,
                             de_vocab_size, encoder_model, decoder_model)

    model.fit_generator(train_generator, steps_per_epoch=steps,
                        callbacks=[bleu_logger])
    model.save_weights('baseline.h5')

    results = pd.DataFrame({'loss': bleu_logger.losses, 'bleu': bleu_logger.scores})
    results.to_csv('baseline_res.csv')
