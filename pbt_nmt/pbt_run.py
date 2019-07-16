import numpy as np

from model import define_nmt
from nmt_utils import nmt_train_generator, bleu_score_enc_dec
from pbt import PbtOptimizer
from members import Member
from wmt import load_wmt

if __name__ == '__main__':
    # To make our experiment reproducible
    np.random.seed(42)

    # Loading 40% of our dataset
    (en_train, en_test, en_tokenizer,
     de_train, de_test, de_tokenizer) = load_wmt(split=0.4)

    # Get the tokenizer length
    en_vocab_size, de_vocab_size = len(en_tokenizer), len(de_tokenizer)

    # Setting the batch size to 64, and the validation set size to 3200
    batch_size = 64
    val_size = 3200
    en_train_t, en_train_v = en_train[val_size:], en_train[:val_size]
    de_train_t, de_train_v = de_train[val_size:], de_train[:val_size]

    # setting model parameters
    hidden_size = 96
    embedding_size = 100
    timesteps = 30

    # creating our search space by creating evenly spaced interval
    lr_values = np.linspace(-4.0, -1.0, num=4).tolist()
    # creating our search space by creating evenly spaced interval
    dropout_values = np.linspace(0.1, 0.5, num=5).tolist()
    parameters = dict(lr=lr_values, dropout=dropout_values)

    population_size = 8

    def build_member(lr, dropout):
        '''
        Building baseline model using lr and dropout rates passed
        in terms of parameters

        For each member we pass our parameters, tune the learning rate,
        using BLEU as our performance metric.
        '''
        model, encoder_model, decoder_model = \
            define_nmt(hidden_size, embedding_size, timesteps,
                       en_vocab_size, de_vocab_size, dropout, np.power(10, lr))

        return Member(model, param_names=['lr', 'dropout'], tune_lr=True, use_eval_metric='bleu', custom_metrics={
            'bleu': lambda x, y, _: bleu_score_enc_dec(encoder_model, decoder_model, x, y, batch_size)
        })

    # Number of steps following convention
    steps = en_train.shape[0] // batch_size
    steps_ready = 1000
    eval_every = 100

    # function to yield sentences instead of loading it to the memory
    def generator_fn(x, y, shuffle=True, looping=True):
        return nmt_train_generator(x, y, de_vocab_size, batch_size,
                                   shuffle=shuffle, looping=looping)

    # Creates population of members with the given dict of hyperparameters
    pbt = PbtOptimizer(build_member, population_size,
                       parameters, steps_ready=steps_ready)
    
    # Trains each member and evaluates it every number of steps
    # saving the best model and its weights
    model, results = pbt.train(en_train_t, de_train_t, en_train_v, de_train_v, steps=steps,
                               eval_every=eval_every, generator_fn=generator_fn)

    model.save_weights('pbt_base.h5')
    results.to_csv('pbt_base_res.csv')
