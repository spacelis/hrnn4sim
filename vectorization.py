#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: vectorization.py
Author: Wen Li
Email: spacelis@gmail.com
Github: http://github.com/spacelis
Description:
"""

import string
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

SIMPLE_TOKENIZER = RegexpTokenizer('[a-zA-Z]+|[0-9]+|-|/')
ALPHABET = string.ascii_lowercase + string.ascii_uppercase + '-/' + string.digits

# pylint: disable=invalid-name

def random_cycled(df):
    ''' Circling through the df with shuffles for each epoch
    '''
    while True:
        s = df.sample(frac=1)
        for i in s.itertuples():
            yield tuple(i)[1:]


def packed(it, columns, size=100):
    ''' Packing iterated items into a package'''
    while True:
        pack = []
        for _, i in zip(range(size), it):
            pack.append(i)
        yield pd.DataFrame.from_records(pack, columns=columns)

def vec_align(arrs):
    ''' Properly padding the nested arrays (tensors)'''
    maxlens = tuple()
    for e in arrs:
        if isinstance(e, (list, tuple)):
            lens = vec_align(e)
            maxlens = tuple(
                max(a, b)
                for a, b in zip(lens, maxlens + ((0,) * (len(lens) - len(maxlens)))))
    return (len(arrs), *maxlens)


def dataset_tokenize(examples, tokenizer=SIMPLE_TOKENIZER):
    ''' Tokenize the dataset with the given tokenizer. '''
    seqa = examples['seqa'].map(tokenizer.tokenize)
    seqb = examples['seqb'].map(tokenizer.tokenize)
    return pd.DataFrame({'seqa': seqa, 'seqb': seqb, 'matched': examples['matched']})


class SubWordVectorizer(object):
    """ Make nested vectors from a word sequence"""
    def __init__(self, alphabet=ALPHABET):
        super(SubWordVectorizer, self).__init__()
        self.alphabet = alphabet
        self.lookup = {k: i for i, k in enumerate(alphabet, 1)}

    def encode_tokenseq(self, seq):
        """Encode a token sequence as a vector of vectors.

        :tokenseq: a list of strings
        :returns: a list of lists of integers

        """
        return [[self.lookup[c] for c in token] for token in seq]

    def vectorize(self, seqs):
        """ Vectorize a set of token seqs

        :seqs: a list of token seqs (string lists)
        :returns: a tensor of encoded seqs [batch_size, max(seq_size), max(token_size)]

        """
        encoded_seqs = [self.encode_tokenseq(s) for s in seqs]
        shape = vec_align(encoded_seqs)
        A = np.zeros(shape)
        for i, seq in enumerate(encoded_seqs):
            for j, token in enumerate(seq):
                for k, c in enumerate(token):
                    A[i, j, k] = c
        return A


class WordVectorizer(object):
    """ Make a vector from a word sequence"""
    def __init__(self, vocab):
        super(WordVectorizer, self).__init__()
        self.vocab = vocab

    @classmethod
    def from_tokens(cls, token_iter):
        ''' Create a vectorizer from a token iterator '''
        return cls({w: i for i, w in enumerate(frozenset(t for t in token_iter), 1)})

    def vectorize(self, seqs):
        """ Vectorize a set of token seqs

        :seqs: a list of token seqs (string lists)
        :returns: a tensor of encoded seqs [batch_size, max(seq_size)]

        """
        encoded_seqs = [[self.vocab[w] for w in seq] for seq in seqs]
        shape = vec_align(encoded_seqs)
        A = np.zeros(shape)
        for i, seq in enumerate(encoded_seqs):
            for j, token in enumerate(seq):
                A[i, j] = token
        return A


# pylint: disable=too-many-arguments
def get_minibatches(dataset, vectorizer, size=100, tokenizer=dataset_tokenize,
                    with_original=False, with_y=True):
    ''' Iterate though the dataset and yielding minibatches.'''
    for ex_orig in packed(random_cycled(dataset), list(dataset), size):
        ex = tokenizer(ex_orig)
        a = vectorizer.vectorize(tuple(ex['seqa']))
        b = vectorizer.vectorize(tuple(ex['seqb']))
        pack = ([a, b], )
        if with_y:
            pack += (ex['matched'], )
        if with_original:
            pack += (ex_orig,)
        if len(pack) == 1:
            yield pack[0]
        else:
            yield pack

def get_fullbatch(dataset, vectorizer, multiple=100, **kargs):
    ''' Return fully vecorized batch '''
    return next(get_minibatches(dataset, vectorizer, len(dataset) // multiple * multiple, **kargs))
