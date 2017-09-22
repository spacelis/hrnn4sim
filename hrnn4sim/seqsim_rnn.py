#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a basic RNN implementation of address matching network using LSTM cells.
"""

# pylint: disable=invalid-name

from itertools import chain

from keras.layers.core import K
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers.recurrent import LSTM

from .vectorization import WordVectorizer, dataset_tokenize
from .base import ModelBase

class SeqSimRNN(ModelBase):
    """ Similarity models based on RNN. """
    def __init__(self):
        super(SeqSimRNN, self).__init__()
        self.state_size = 256
        self.feature_size = 256

    def build(self):
        ''' Build a RNN based model. '''
        K.set_session(self.session)
        A = Input(shape=(None,))
        B = Input(shape=(None,))
        em = Embedding(len(self.vectorizer.vocab) + 1, self.state_size, mask_zero=True)
        emA = em(A)
        emB = em(B)

        fetures = Dense(self.state_size, activation='relu')
        fA = fetures(emA)
        fB = fetures(emB)

        lstm1 = LSTM(self.state_size,
                     dropout=0.2, recurrent_dropout=0.2, return_state=True)(fA)
        lstm2 = LSTM(self.state_size,
                     dropout=0.2, recurrent_dropout=0.2)(fB, initial_state=lstm1[1:])
        output = Dense(1, activation='sigmoid')(lstm2)
        self.model = Model(inputs=[A, B], outputs=[output])


        ### Compile the models by supplying a loss funciton and an optimizer.
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def make_vectorizer(self, examples):
        examples = dataset_tokenize(examples)
        return WordVectorizer.from_tokens(
            t
            for s in chain(iter(examples['seqa']), iter(examples['seqb']))
            for t in s
        )
