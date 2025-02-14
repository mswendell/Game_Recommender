#!/usr/bin/env python3
"""
Train a recommendation model.

Usage:
    train-model [options]

Options:
    --hybrid
        Use the hybrid tag model.
    --als
        Use the ALS matrix factorization model.
    --popular
        Use the popular model.
    --bias
        Use the bias model.
    --logistic
        Use logistic loss instead of BPR.
    --mse
        Use MSE loss instead of BPR.
    -d DEV, --device=DEV
        Use DEV for training.
    -n NAME
        Save as model NAME
    -e EPOCHS
        Train for EPOCHS epochs [default: 5]
    -k FEATURES
        Use FEATURES latent features [default: 50]
    -r REG
        Use REG regularization [default: 0.01]
    -c CONFIG
        Use config from FILE
    --verbose
        Output verbose debugging information.
"""

from pathlib import Path
import sys
import logging
from docopt import docopt
import tomli

from binpickle import dump, codecs
import pandas as pd

from hygame import GameTagMF
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import Popular
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.bias import Bias

_log = logging.getLogger('train-model')


def load_data():
    ratings = pd.read_parquet('data/a3-train-actions.parquet', columns=['user', 'item'])
    _log.info('loaded %d ratings (%d bytes)', len(ratings),
              ratings.memory_usage(deep=True).sum())

    devs = pd.read_parquet('data/devs.parquet')
    _log.info('loaded devs\n%s', devs)

    genres = pd.read_parquet('data/genres.parquet')
    _log.info('loaded genres\n%s', genres)

    return ratings, devs, genres


def mf_opts(opts, name):
    if opts['-c']:
        _log.info('using section %s from %s', name, opts['-c'])
        with open(opts['-c'], 'rb') as rf:
            config = tomli.load(rf)
        cfg = config[name]
        if 'reg' in cfg and isinstance(cfg['reg'], list):
            cfg['reg'] = tuple(cfg['reg'])
        return cfg
    else:
        _log.info('configuring from command line')
        reg = opts['-r']
        if ',' in reg:
            reg = tuple(float(r) for r in reg.split(','))
        else:
            reg = float(reg)
        return {
            'epochs': int(opts['-e']),
            'features': int(opts['-k']),
            'reg': reg
        }


def build_hybrid(opts):
    config = mf_opts(opts, opts.get('-n', 'hybrid'))
    if opts['--logistic']:
        loss = 'logistic'
    elif opts['--mse']:
        loss = 'mse'
    else:
        loss = 'bpr'

    device = opts.get('--device', None)

    components = config.get('components', 'all')
    algo = GameTagMF(config['features'], reg=config['reg'], epochs=config['epochs'], components=components, loss=loss, device=device)
    algo = Recommender.adapt(algo)
    return algo


def build_pop(opts):
    algo = Popular()
    return algo

def build_bias(opts):

    algo = Bias()
    return algo

def build_als(opts):
    config = mf_opts(opts, opts.get('-n', 'als'))

    algo = ImplicitMF(config['features'], reg=config['reg'])
    return algo


def main(opts):
    ratings, devs, genres = load_data()
    if opts['--hybrid']:
        model = build_hybrid(opts)
    elif opts['--als']:
        model = build_als(opts)
    elif opts['--popular']:
        model = build_pop(opts)
    elif opts['--bias']:
        model = build_bias(opts)
        ratings = pd.read_parquet('data/a3-train-actions.parquet', columns=['user', 'item','rating'])
    else:
        _log.error('no model specified')
        sys.exit(1)

    model = Recommender.adapt(model)
    _log.info('training %s', model)
    model.fit(ratings, genres=genres, devs=devs)
    
    mdir = Path('models')
    mdir.mkdir(exist_ok=True)
    name = opts['-n']
    mfile = mdir / f'{name}.bpk'
    _log.info('saving model to %s', mfile)
    dump(model, mfile, codec=codecs.Blosc())


if __name__ == '__main__':
    opts = docopt(__doc__)
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    logging.getLogger('numba').setLevel(logging.INFO)

    main(opts)
