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

import click


@click.command()
@click.option('-m', '--model', default='HRNN',
              help='HRNN or RNN for encoder/decoder (default: HRNN)')
@click.argument('csvfile', type=click.Path(exists=True))
def console(model, csvfile):
    ''' Train a model for similarity measures.
    '''
    from hrnn4sim.seqsim_hrnn import SeqSimHRNN
    from hrnn4sim.seqsim_rnn import SeqSimRNN
    if model == 'HRNN':
        model_class = SeqSimHRNN
    elif model == 'RNN':
        model_class = SeqSimRNN
    else:
        print('Error: {model} is not recognized as a model. Please use HRNN or RNN.')
    mdl = model_class()
    mdl.train(csvfile)


if __name__ == "__main__":
    console()  # pylint: disable=no-value-for-parameter
