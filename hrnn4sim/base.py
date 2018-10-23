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


def read_data(fin, filename):
    """ Resove file format for the input file and return a file object """
    if filename.endswith('.csv'):
        return pd.read_csv(fin)
    elif filename.endswith('.feather'):
        return pd.read_feather(filename)
    raise ValueError(f'File format not supported: {filename}')


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
        """ Loading model from files """
        model_path = pjoin(job_dir, model_dir, 'model_{}.h5'.format(model_label))
        with file_io.FileIO(model_path, mode='rb') as fin:
            with file_io.FileIO('model.h5.tmp', mode='wb') as fout:
                fout.write(fin.read())
        self.model.load_weights('model.h5.tmp')
        print("Load {}".format(model_path))


    def train(self, trainfile, model_label=None,   # pylint: disable=too-many-arguments
              epochs=30, batch_size=100,
              val_file=None, val_split=0.8,
              shuffle=False, include_eos=False,
              job_dir='.', model_dir='ckpt'):
        # pylint: disable=too-many-locals
        ''' Train the model '''
        with file_io.FileIO(trainfile, 'r') as fin:
            examples = read_data(fin, trainfile)
        if shuffle:
            examples = examples.sample(frac=1).reset_index(drop=True)
        else:
            examples = examples.reset_index(drop=True)
        if val_file is not None:
            with file_io.FileIO(trainfile, 'r') as fin:
                val_examples = read_data(fin, trainfile)
            if shuffle:
                val_examples = val_examples.sample(frac=1).reset_index(drop=True)
            else:
                val_examples = val_examples.reset_index(drop=True)
            self.vectorizer = self.make_vectorizer(pd.concat([examples, val_examples]),
                                                   include_eos=include_eos)
        else:
            self.vectorizer = self.make_vectorizer(examples, include_eos=include_eos)
        self.build()

        if model_label is not None:
            self.load_model(job_dir, model_dir, model_label)

        label = '{}_{}'.format(self.__class__.__name__, datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Write Summaries to Tensorboard log
        tensorboardCB = TensorBoard(
            log_dir=pjoin(job_dir, 'tfgraph', label),
            #histogram_freq=100,
            write_graph=True)

        ckpt_label = '{}_epoch_{{epoch:02d}}_acc_{{val_acc:.4f}}'.format(label)
        checkpointCB = ModelCheckpoint(ckpt_label, monitor='val_acc', save_weights_only=True)

        # Train the model
        if val_file is not None:
            train_set, valid_set = self.split_examples(examples, val_split)
        else:
            train_set, valid_set = examples, val_examples
        x, y = get_fullbatch(train_set, self.vectorizer, multiple=batch_size)
        vx, vy = get_fullbatch(valid_set, self.vectorizer, multiple=batch_size)

        # Training

        K.set_session(self.session)
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs,
                       validation_data=(vx, vy),
                       callbacks=[tensorboardCB, checkpointCB])
        # Validation
        loss, acc = self.model.evaluate(vx, vy, batch_size=batch_size)

        model_label = '{}_loss_{:.4f}_acc_{:.4f}'.format(label, loss, acc)
        self.save_model(job_dir, model_dir, model_label)

        print()
        print('Loss =', loss)
        print('Accuracy =', acc)

    def test(self, testfile, model_label, batch_size=100,  # pylint: disable=too-many-arguments
             include_eos=False, job_dir='.', model_dir='ckpt'):
        """ Evaluate model on the test data """
        with file_io.FileIO(testfile, 'r') as fin:
            examples = read_data(fin, testfile)
        self.vectorizer = self.make_vectorizer(examples, include_eos=include_eos)
        self.build()
        K.set_session(self.session)
        self.load_model(job_dir, model_dir, model_label)

        x, y = get_fullbatch(examples, self.vectorizer, multiple=batch_size)
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
