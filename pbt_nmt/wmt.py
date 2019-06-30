import os

import numpy as np

from data_utils import load_nmt


def load_wmt(data_folder='data', maxlen=30, split=0.5):
    path = os.path.join(data_folder, 'europarl.npz')
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as data:
            en_train, en_test, en_tokenizer = data['en_train'], data['en_test'], data['en_tokenizer']
            de_train, de_test, de_tokenizer = data['de_train'], data['de_test'], data['de_tokenizer']
            return en_train, en_test, en_tokenizer.item(), de_train, de_test, de_tokenizer.item()

    else:
        en_file = 'europarl-v7.de-en.en'
        de_file = 'europarl-v7.de-en.de'

        en_train, en_test, en_tokenizer, de_train, de_test, de_tokenizer = load_nmt(
            src_path=os.path.join(os.getcwd(), 'pbt_nmt', 'data', en_file),
            tar_path=os.path.join(os.getcwd(), 'pbt_nmt', 'data', de_file),
            maxlen=maxlen, split=split, seed=0)

        data = [en_train, en_test, en_tokenizer, de_train, de_test, de_tokenizer]
        np.savez(path, en_train=en_train, en_test=en_test, en_tokenizer=en_tokenizer,
                 de_train=de_train, de_test=de_test, de_tokenizer=de_tokenizer)

    return data
