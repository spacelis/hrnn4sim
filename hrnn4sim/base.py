#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: training.py
Author: Wen Li
Email: spacelis@gmail.com
Github: http://github.com/spacelis
Description: Training utility functions
"""

# pylint: disable=invalid-name

from datetime import datetime

import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

from .vectorization import get_fullbatch, get_minibatches
from .vectorization import dataset_tokenize


class ModelBase(object):
    """ A Base model for handling training, validation and prediction"""
    def __init__(self):
        super(ModelBase, self).__init__()
        self.model = None
        self.vectorizer = None
        self.session = tf.Session()

    def split_examples(self, examples):  # pylint: disable=no-self-use
        ''' Split training and validating data set '''

        total_cnt = len(examples)
        train_cnt = total_cnt // 10 * 8
        valid_cnt = total_cnt - train_cnt

        train_set = examples[:train_cnt]
        valid_set = examples[-valid_cnt:]
        return train_set, valid_set

    def make_vectorizer(self, examples):
        ''' Make a vectorizer for the model '''
        raise NotImplementedError()

    def build(self):
        ''' Build the model '''
        raise NotImplementedError()

    def train(self, filename, epochs=30):
        ''' Train the model '''
        examples = pd.read_csv(filename)
        self.vectorizer = self.make_vectorizer(examples)
        self.build()

        label = f'{self.__class__.__name__}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        # Write Summaries to Tensorboard log
        tbCallBack = TensorBoard(
            log_dir=f'./tfgraph/{label}',
            histogram_freq=100, write_graph=True)

        # Save the model and parameters
        ckpCallBack = ModelCheckpoint(
            f'./ckpt/model_{label}.ckpt',
            monitor='acc', save_best_only=True, save_weights_only=True, mode='max')

        # Train the model
        train_set, valid_set = self.split_examples(examples)
        x, y = get_fullbatch(train_set, self.vectorizer)
        vx, vy = next(get_minibatches(valid_set, self.vectorizer, 1000, with_original=False))

        # Training

        K.set_session(self.session)
        self.model.fit(x, y, batch_size=100, epochs=epochs,
                       callbacks=[tbCallBack, ckpCallBack])
        # Validation
        loss, acc = self.model.evaluate(vx, vy, batch_size=100)
        print()
        print('Loss =', loss)
        print('Accuracy =', acc)

    def predict(self, items):
        ''' Predict the the matching of the items '''
        x = get_fullbatch(dataset_tokenize(items), self.vectorizer, with_y=False)

        K.set_session(self.session)
        pred = self.model.predict(x, batch_size=100)
        return pd.DataFrame({
            'seqa': items['seqa'],
            'seqb': items['seqb'],
            'matched': pred,
        })
