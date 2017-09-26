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

from __future__ import print_function
from datetime import datetime
from os.path import join as pjoin

import pandas as pd
import tensorflow as tf
from tensorflow.python.lib.io import file_io
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

    def train(self, filename, epochs=30, job_dir='.'):
        # pylint: disable=too-many-locals
        ''' Train the model '''
        with file_io.FileIO(filename, 'r') as fin:
            examples = pd.read_csv(fin)
        self.vectorizer = self.make_vectorizer(examples)
        self.build()

        label = '{}_{}'.format(self.__class__.__name__, datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Write Summaries to Tensorboard log
        tbCallBack = TensorBoard(
            log_dir=pjoin(job_dir, 'tfgraph', label),
            #histogram_freq=100,
            write_graph=True)

        # Train the model
        train_set, valid_set = self.split_examples(examples)
        x, y = get_fullbatch(train_set, self.vectorizer)
        vx, vy = next(get_minibatches(valid_set, self.vectorizer, 1000, with_original=False))

        # Training

        K.set_session(self.session)
        self.model.fit(x, y, batch_size=100, epochs=epochs,
                       callbacks=[tbCallBack])
        # Validation
        loss, acc = self.model.evaluate(vx, vy, batch_size=100)
        print()
        self.model.save_weights('model.h5.tmp')
        with file_io.FileIO('model.h5.tmp', mode='r') as fin:
            model_path = pjoin(job_dir, 'ckpt', 'model_{}.h5'.format(label))
            with file_io.FileIO(model_path, mode='w') as fout:
                fout.write(fin.read())
                print("Saved {}".format(model_path))
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
