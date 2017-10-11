#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
We apply the SeqSimRNN and SeqSimHRNN to address matching by consider
an textual address as a seqence of words which are also seqences of
alphanums. The motivation is that the words in addresses are more likely
to be unique in the sense that they are created for representing a name
via combining, abbreviating, mixing of words. Treating each word as
distinuishable especially for numbers may limit the ability for the
models to generalize.
"""

from __future__ import print_function
import sys
import click


@click.command()
@click.option('-m', '--model', default='HRNN',
              help='HRNN or RNN for encoder/decoder (default: HRNN)')
@click.option('--job-dir', default='.',
              help='The directory to use for the job, e.g., models, logs. (default: .)')
@click.option('-e', '--epochs', default=2,
              help='Number of epochs to run the training. (default: 2)')
@click.option('-b', '--batch-size', default=100,
              help='The size of the batch. (default: 100)')
@click.option('-s', '--embedding-size', default=64,
              help='The size of the embedding states (HRNN only). (default: 64)')
@click.option('-t', '--split-ratio', default=0.8,
              help='The ratio for train/test split. (default: 0.8)')
@click.option('-h', '--state-size', default=256,
              help='The size of the hidden states. (default: 64)')
@click.option('-l', '--log-device', is_flag=True)
@click.argument('csvfile')
def console(model, job_dir, epochs, batch_size, split_ratio,
            embedding_size, state_size, log_device, csvfile):
    ''' Train a model for similarity measures.
    '''
    from hrnn4sim.seqsim_hrnn import SeqSimHRNN
    from hrnn4sim.seqsim_bihrnn import SeqSimBiHRNN
    from hrnn4sim.seqsim_rnn import SeqSimRNN
    if model == 'HRNN':
        mdl = SeqSimHRNN(embedding_size=embedding_size, state_size=state_size,
                         batch_size=batch_size, log_device=log_device)
    elif model == 'BHRNN':
        mdl = SeqSimBiHRNN(embedding_size=embedding_size, state_size=state_size,
                           batch_size=batch_size, log_device=log_device)
    elif model == 'RNN':
        mdl = SeqSimRNN(state_size=state_size, log_device=log_device)
    else:
        print('Error: {model} is not recognized as a model. Please use HRNN or RNN.')
        sys.exit(1)
    mdl.train(csvfile, epochs=epochs, batch_size=batch_size,
              split_ratio=split_ratio, job_dir=job_dir)


if __name__ == "__main__":
    console()  # pylint: disable=no-value-for-parameter
