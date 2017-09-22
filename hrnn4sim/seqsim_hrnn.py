#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hieararchical RNN can is designed for capture sequence embeddings in more than
one level. The paper introducing HRNN is for encoding paragraphs via encoded
sentences. That is a paragraph of text can be seen as (sentence) sequences of
(word) sequences. This can be generalized to letter-word-senstence hierarchy.
"""

# pylint: disable=invalid-name

from keras.layers.core import K
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, SimpleRNN

from .vectorization import SubWordVectorizer
from .base import ModelBase


class SeqSimHRNN(ModelBase):
    """ Similiarity Model based on HierarchicalRNN """
    def __init__(self, embedding_size=64, state_size=256):  # pylint: disable=too-many-locals
        super(SeqSimHRNN, self).__init__()
        self.embedding_size = embedding_size
        self.state_size = state_size

    def build(self):
        ''' Build the model '''
        K.set_session(self.session)
        ### Hierarchical Recurrent Neural Network
        A = Input(batch_shape=(100, None, None))
        B = Input(batch_shape=(100, None, None))
        masking = Lambda(
            lambda inputs: K.any(
                K.not_equal(
                    K.cast(
                        K.any(
                            K.not_equal(inputs, 0.), axis=-1
                        ), K.floatx()
                    ), 0.
                ), axis=-1, keepdims=True
            )
        )


        em = Embedding(len(self.vectorizer.alphabet) + 1, self.embedding_size, mask_zero=True)
        encoder = SimpleRNN(self.state_size, dropout=0.2, recurrent_dropout=0.2)
        features = Dense(self.state_size)
        seq_encoder = Lambda(lambda inputs: features(encoder(em(inputs))))
        encoderl = TimeDistributed(seq_encoder)
        codA = encoderl(A, mask=masking(A))
        codB = encoderl(B, mask=masking(B))

        lstm1 = LSTM(self.state_size,
                     dropout=0.2, recurrent_dropout=0.2, return_state=True)(codA)
        lstm2 = LSTM(self.state_size,
                     dropout=0.2, recurrent_dropout=0.2)(codB, initial_state=lstm1[1:])
        output = Dense(1, activation='sigmoid')(lstm2)
        self.model = Model(inputs=[A, B], outputs=[output])

        ### Compile the models by supplying a loss funciton and an optimizer.
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def make_vectorizer(self, examples):  #pylint: disable=unused-variable
        ''' Make a vectorizer for HRNN '''
        return SubWordVectorizer()
