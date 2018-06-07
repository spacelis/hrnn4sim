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
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, SimpleRNN

from .vectorization import SubWordVectorizer
from .base import ModelBase


class SeqSimBiHRNN(ModelBase):
    """ Similiarity Model based on HierarchicalRNN """
    def __init__(self, embedding_size=64, state_size=100,   # pylint: disable=too-many-arguments
                 batch_size=100,
                 wenc_dor=0.5, senc_dor=0.5, **kwargs):
        super(SeqSimBiHRNN, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.wenc_dor = wenc_dor
        self.senc_dor = senc_dor

    def build(self):  # pylint: disable=too-many-locals
        ''' Build the model '''
        K.set_session(self.session)
        ### Hierarchical Recurrent Neural Network
        A = Input(batch_shape=(self.batch_size, None, None))
        Ar = Input(batch_shape=(self.batch_size, None, None))
        B = Input(batch_shape=(self.batch_size, None, None))
        Br = Input(batch_shape=(self.batch_size, None, None))
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
        encoder = SimpleRNN(self.state_size, dropout=self.wenc_dor, recurrent_dropout=self.wenc_dor)
        features = Dense(self.state_size)
        seq_encoder = Lambda(lambda inputs: features(encoder(em(inputs))))
        encoder_l = TimeDistributed(seq_encoder)
        encoder_r = TimeDistributed(seq_encoder)

        codA = encoder_l(A, mask=masking(A))
        codB = encoder_l(B, mask=masking(B))
        codAr = encoder_r(Ar, mask=masking(Ar))
        codBr = encoder_r(Br, mask=masking(Br))

        lstmA = LSTM(self.state_size, dropout=self.senc_dor, recurrent_dropout=self.senc_dor,
                     return_state=True)(codA)
        lstmAr = LSTM(self.state_size, dropout=self.senc_dor, recurrent_dropout=self.senc_dor,
                      return_state=True)(codAr)

        concat_mem = Concatenate(axis=1)([lstmA[-2], lstmAr[-2]])
        concat_state = Concatenate(axis=1)([lstmA[-1], lstmAr[-1]])
        merged_mem = Dense(self.state_size, activation='relu')(concat_mem)
        merged_state = Dense(self.state_size, activation='relu')(concat_state)

        lstmB = LSTM(self.state_size, dropout=self.senc_dor, recurrent_dropout=self.senc_dor)\
            (codB, [merged_mem, merged_state])
        lstmBr = LSTM(self.state_size, dropout=self.senc_dor, recurrent_dropout=self.senc_dor)\
            (codBr, [merged_mem, merged_state])
        combined = Concatenate(axis=1)([lstmB, lstmBr])
        output = Dense(1, activation='sigmoid')(combined)
        self.model = Model(inputs=[A, Ar, B, Br], outputs=[output])

        ### Compile the models by supplying a loss funciton and an optimizer.
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model.summary()

    def make_vectorizer(self, examples, **kwargs):  #pylint: disable=unused-variable
        ''' Make a vectorizer for HRNN '''
        return SubWordVectorizer(bidir=True, **kwargs)
