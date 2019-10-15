from collections import Counter

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence


def train_test_split(data, split, seed=None):
    ''' splitting data given into train and test '''
    n_total = len(data)
    n_train = int(n_total * split)
    np.random.seed(seed)
    index = np.random.permutation(n_total)
    train_index, test_index = index[:n_train], index[n_train:]
    train = [data[i] for i in train_index]
    test = [data[i] for i in test_index]
    return train, test


def read_texts(path):
    with open(path, encoding='utf-8', mode='r') as file:
        return list(file)


def texts_to_words(texts):
    ''' Tokenizing sentences into words by removing numbers, punctuations '''
    filters = '0123456789!"„“#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    return [[w.replace("'", '') for w in text_to_word_sequence(t, filters=filters)] for t in texts]


def fit_encode_texts(train, test, is_source, maxlen):
    ''' Starts with tokenizing the sentence
    and eliminating any word with freq of occurrence less than 5 times,
    building our vocabulary - based on unique words of our training data -
    Changing the words to their corresponding numbers as RNN can't
    deal with words
    
    Finally it returns the train & test data as well as the tokenizer '''
    train = texts_to_words(train)

    counter = Counter(w for t in train for w in t)
    vocab = set([w for w, c in counter.items() if c >= 5])
    tokenizer = Tokenizer([[w for w in t if w in vocab] for t in train])

    padding_type = 'pre' if is_source else 'post'
    train = tokenizer.texts_to_sequences(train, maxlen, padding_type)
    test = tokenizer.texts_to_sequences(test, maxlen, padding_type)

    return train, test, tokenizer


def load_nmt(src_path, tar_path, maxlen, split, seed):
    """
    We read dataset from the given paths,
    followed by splitting them based on the split param,
    Tokenizing sentences as well as cleaning them
    Trim and pad the sentences to have a maxlen 
    """
    src_text = read_texts(src_path)
    tar_text = read_texts(tar_path)

    src_train_text, src_test_text = train_test_split(src_text, split, seed)
    tar_train_text, tar_test_text = train_test_split(tar_text, split, seed)

    src_train, src_test, src_tokenizer = fit_encode_texts(
        src_train_text, src_test_text, is_source=True, maxlen=maxlen)
    tar_train, tar_test, tar_tokenizer = fit_encode_texts(
        tar_train_text, tar_test_text, is_source=False, maxlen=maxlen)

    return src_train, src_test, src_tokenizer, tar_train, tar_test, tar_tokenizer


class Tokenizer:
    ''' Tokenizer class  '''
    PAD, PAD_TOK = 0, '<pad>'
    UNK, UNK_TOK = 1, '<unk>'
    BOS, BOS_TOK = 2, '<bos>'

    def __init__(self, texts):
        self.word2idx = {}
        self.idx2word = []

        for word in [Tokenizer.PAD_TOK, Tokenizer.UNK_TOK, Tokenizer.BOS_TOK]:
            self._add_word(word)

        for text in texts:
            for word in text:
                self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2idx:
            idx = len(self)
            self.idx2word.append(word)
            self.word2idx[word] = idx

    def __len__(self):
        return len(self.idx2word)

    def texts_to_sequences(self, texts, maxlen, padding_type):
        if isinstance(texts[0], str):
            texts = texts_to_words(texts)

        seqs = [[Tokenizer.BOS] +
                [self.word2idx.get(word, Tokenizer.UNK) for word in text] for text in texts]
        seqs = pad_sequences(seqs, maxlen=maxlen,
                             padding=padding_type, truncating='post')
        return seqs

    def sequences_to_texts(self, seqs, as_str=True):
        texts = []
        for seq in seqs:
            seq = np.trim_zeros(seq)
            if seq[0] == Tokenizer.BOS:
                seq = seq[1:]
            text = [self.idx2word[idx] for idx in seq]
            if as_str:
                text = ' '.join(text)
            texts.append(text)
        return texts
