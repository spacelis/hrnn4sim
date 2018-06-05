#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: data_augmentation.py
Author: Wen Li
Email: spacelis@gmail.com
Github: http://github.com/spacelis
Description: Augmenting data with some sythetic negative examples.
"""

# pylint: disable=invalid-name

from __future__ import print_function
import sys
import re
import random

import pandas as pd
import click

## Data augmentation

### Some unility functions for data augmentations
def rand_delta(m):
    ''' Generate a random numbers by applying a random delta'''
    x = int(m.group(0))
    y = random.randint(1, x + 20)
    if x == y:
        return str(x + 1)
    return str(y)


def change_num(addr):
    ''' Change the add by applying a random delta to the numbers'''
    return re.sub('[0-9]+', rand_delta, addr)


def get_neg_examples(df):
    ''' Generate negative examples '''
    addrPoolA = list(frozenset(df['addra']))
    sampleA = random.sample(addrPoolA, len(addrPoolA))[:len(addrPoolA)//2 * 2]
    exA = sampleA[0:len(sampleA):2], sampleA[1:len(sampleA):2]

    addrPoolB = list(frozenset(df['addrb']))
    sampleB = random.sample(addrPoolB, len(addrPoolB))[:len(addrPoolB)//2 * 2]
    exB = sampleB[0:len(sampleB):2], sampleB[1:len(sampleB):2]

    exC = [], []
    for addr in sampleA:
        cn_addr = change_num(addr)
        if cn_addr != addr:
            exC[0].append(addr)
            exC[1].append(change_num(addr))

    exD = [], []
    for addr in sampleB:
        cn_addr = change_num(addr)
        if cn_addr != addr:
            exD[0].append(addr)
            exD[1].append(change_num(addr))

    return pd.DataFrame({'addra': exA[0] + exB[0] + exC[0] + exD[0],
                         'addrb': exA[1] + exB[1] + exC[1] + exD[1]})


def get_pos_examples(df):
    ''' Make some more positive examples by cloning addresses '''
    addrPoolA = list(frozenset(df['addra']))
    addrPoolB = list(frozenset(df['addrb']))
    return pd.DataFrame({'addra': list(df['addra']) + addrPoolA + addrPoolB,
                         'addrb': list(df['addrb']) + addrPoolA + addrPoolB})

def data_augmentation(df):
    ''' Data augmentation via constructing negative examples
        :param df: A pandas dataframe having columns of (addra, addrb, matched)
    '''
    neg = get_neg_examples(df)
    pos = get_pos_examples(df)
    pos.loc[:, 'matched'] = 1
    neg.loc[:, 'matched'] = 0
    return pd.concat([pos, neg]).rename(columns={
        'addra': 'seqa',
        'addrb': 'seqb'
    })


@click.command()
@click.argument('src', type=click.Path(exists=True))
@click.argument('dst', type=click.Path(exists=False))
def console(src, dst):
    ''' This tool is for creating a augmented data set for training models for
        address matchings.

        The expected input is a CSV file of positive examples with
        the headers (addra, addrb). The output will be a CSV file of
        table filled with augmented data with the headers (seqa, seqb,
        matched). Augmentation includes number changing, identity
        matching.
    '''
    if src.endswith('csv'):
        raw_data = pd.read_csv(src)
    elif src.endswith('.feather'):
        raw_data = pd.read_feather(src)
    else:
        print('Error: Input file format not supported', file=sys.stderr)
        sys.exit(-1)

    print("uniqA={}".format(raw_data['addra'].nunique()))
    print("uniqB={}".format(raw_data['addrb'].nunique()))
    print("pairCnt={}".format(len(raw_data)))

    examples = data_augmentation(raw_data).sample(frac=1)  # Randomized rows
    print(examples.head())
    if src.endswith('csv'):
        examples.to_csv(dst, index=False)
    elif src.endswith('.feather'):
        examples.reset_index()[['seqa', 'seqb', 'matched']].to_feather(dst)
    else:
        print('Error: Output file format not supported', file=sys.stderr)
        sys.exit(-1)

if __name__ == "__main__":
    console()  # pylint: disable=no-value-for-parameter
