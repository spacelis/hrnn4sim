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
from keras.callbacks import TensorBoard

from .vectorization import get_fullbatch, get_minibatches
from .vectorization import dataset_tokenize


class ModelBase(object):
    """ A Base model for handling training, validation and prediction"""
    def __init__(self, log_device=False):
        super(ModelBase, self).__init__()
        self.model = None
        self.vectorizer = None
        if log_device:
            self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        else:
            self.session = tf.Session()

    def split_examples(self, examples, ratio=0.8):  # pylint: disable=no-self-use
        ''' Split training and validating data set '''

        total_cnt = len(examples)
        train_cnt = int(total_cnt * ratio)
        valid_cnt = total_cnt - train_cnt

        train_set = examples[:train_cnt]
        valid_set = examples[-valid_cnt:]
        return train_set, valid_set

    def make_vectorizer(self, examples, **kwargs):
        ''' Make a vectorizer for the model '''
        raise NotImplementedError()

    def build(self):
        ''' Build the model '''
        raise NotImplementedError()

    def save_model(self, job_dir, model_dir, model_label):
        """ Save the trained model to the job_dir"""
        self.model.save_weights('model.h5.tmp')
        with file_io.FileIO('model.h5.tmp', mode='rb') as fin:
            model_path = pjoin(job_dir, model_dir, 'model_{}.h5'.format(model_label))
            with file_io.FileIO(model_path, mode='wb') as fout:
                fout.write(fin.read())
                print("Saved {}".format(model_path))

    def load_model(self, job_dir, model_dir, model_label):
        model_path = pjoin(job_dir, model_dir, 'model_{}.h5'.format(model_label))
        with file_io.FileIO(model_path, mode='rb') as fin:
            with file_io.FileIO('model.h5.tmp', mode='wb') as fout:
                fout.write(fin.read())
        self.model.load_weights('model.h5.tmp')
        print("Load {}".format(model_path))


    def train(self, filename, epochs=30, batch_size=100,
              split_ratio=0.8, include_eos=False,
              job_dir='.', model_dir='ckpt'):
        # pylint: disable=too-many-locals
        ''' Train the model '''
        with file_io.FileIO(filename, 'r') as fin:
            examples = pd.read_csv(fin)
        self.vectorizer = self.make_vectorizer(examples, include_eos=include_eos)
        self.build()

        label = '{}_{}'.format(self.__class__.__name__, datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Write Summaries to Tensorboard log
        tbCallBack = TensorBoard(
            log_dir=pjoin(job_dir, 'tfgraph', label),
            #histogram_freq=100,
            write_graph=True)

        # Train the model
        train_set, valid_set = self.split_examples(examples, split_ratio)
        x, y = get_fullbatch(train_set, self.vectorizer, multiple=batch_size)
        vx, vy = next(get_minibatches(valid_set, self.vectorizer, batch_size, with_original=False))

        # Training

        K.set_session(self.session)
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs,
                       callbacks=[tbCallBack])
        # Validation
        loss, acc = self.model.evaluate(vx, vy, batch_size=batch_size)

        model_label = '{}_loss_{:.4f}_acc_{:.4f}'.format(label, loss, acc)
        self.save_model(job_dir, model_dir, model_label)

        print()
        print('Loss =', loss)
        print('Accuracy =', acc)

    def test(self, filename, model_label, batch_size=100, include_eos=False,
             job_dir='.', model_dir='ckpt'):
        with file_io.FileIO(filename, 'r') as fin:
            examples = pd.read_csv(fin)
        self.vectorizer = self.make_vectorizer(examples, include_eos=include_eos)
        self.build()
        self.load_model(job_dir, model_dir, model_label)

        x, y = get_fullbatch(examples, self.vectorizer, multiple=batch_size)
        K.set_session(self.session)
        loss, acc = self.model.evaluate(x, y, batch_size=batch_size)
        print()
        print('Loss =', loss)
        print('Accuracy =', acc)

    def predict(self, items, batch_size=100):
        ''' Predict the the matching of the items '''
        x = get_fullbatch(dataset_tokenize(items), self.vectorizer, with_y=False)

        K.set_session(self.session)
        pred = self.model.predict(x, batch_size=batch_size)
        return pd.DataFrame({
            'seqa': items['seqa'],
            'seqb': items['seqb'],
            'matched': pred,
        })
