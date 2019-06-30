import numpy as np

from model import define_nmt
from nmt_utils import nmt_train_generator, bleu_score_enc_dec
from members import Member
from pbt_pso import PbtPsoOptimizer
from wmt import load_wmt

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

    # search space
    lr_values = np.linspace(-4.0, -1.0, num=4).tolist()
    dropout_values = np.linspace(0.1, 0.5, num=5).tolist()
    parameters = dict(lr=lr_values, dropout=dropout_values)

    population_size = 8

    def build_member(lr, dropout):
        model, encoder_model, decoder_model = \
            define_nmt(hidden_size, embedding_size, timesteps,
                       en_vocab_size, de_vocab_size, dropout, np.power(10, lr))

        return Member(model, param_names=['lr', 'dropout'], tune_lr=True, use_eval_metric='bleu', custom_metrics={
            'bleu': lambda x, y, _: bleu_score_enc_dec(encoder_model, decoder_model, x, y, batch_size)
        })

    steps = en_train.shape[0] // batch_size
    steps_ready = 1000
    eval_every = 100

    generator_fn = lambda x, y, shuffle=True, looping=True: nmt_train_generator(x, y, de_vocab_size, batch_size,
                                                                                shuffle=shuffle, looping=looping)
    pbt = PbtPsoOptimizer(build_member, population_size, parameters, steps_ready=steps_ready,
                          omega=0.5, phi1=0.5, phi2=1.0)
    model, results = pbt.train(en_train_t, de_train_t, en_train_v, de_train_v, steps=steps,
                        eval_every=eval_every, generator_fn=generator_fn)

    model.save_weights('pbt_pso.h5')
    results.to_csv('pbt_pso_res.csv')



