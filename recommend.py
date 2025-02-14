#!/usr/bin/env python3
"""
Generate recommendations.

Usage:
    train-model [options] (--eval|--dev) MODEL

Options:
    --eval
        Recommend for eval data.
    --dev
        Recommend for dev data.
    -n N
        Recommend lists of size N [default: 100]
    -j J
        Use J jobs
    --cuda
        Use CUDA for inference.
    --verbose
        Turn on verbose output.
"""

from pathlib import Path
import sys
import os
import logging
from docopt import docopt
import json

from binpickle import load
import pandas as pd

from lenskit import batch
from lenskit import topn
import torch

_log = logging.getLogger('train-model')

data_dir = Path('data')
model_dir = Path('models')
rec_dir = Path('recs')

torch.set_num_interop_threads(1)
torch.set_num_threads(2)

def load_data(name):
    df = data_dir / f'a3-{name}-actions.parquet'
    _log.info('loading %s', df)
    return pd.read_parquet(df)


def load_model(name):
    mf = model_dir / f'{name}.bpk'
    _log.info('loading %s', mf)
    return load(mf)


def main(opts):
    data_name = 'dev' if opts['--dev'] else 'eval'
    model_name = opts['MODEL']
    n = int(opts['-n'])
    n_jobs = opts['-j']
    if n_jobs:
        n_jobs = int(n_jobs)
    
    test = load_data(data_name)
    model = load_model(model_name)

    if opts['--cuda'] and hasattr(model.predictor, 'to'):
        model.predictor.to('cuda')
        n_jobs = 1

    recs = batch.recommend(model, test['user'].unique(), n, n_jobs=n_jobs)

    rec_dir.mkdir(exist_ok=True)
    ofn = rec_dir / f'{model_name}-{data_name}.parquet'
    _log.info('saving to %s', ofn)
    recs.to_parquet(ofn)

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    scores = rla.compute(recs, test[['user', 'item']])
    ndcg = scores['ndcg'].mean()
    _log.info('final nDCG: %.4f', ndcg)

    met_fn = rec_dir / f'{model_name}-{data_name}.json'
    with met_fn.open('w') as fp:
        json.dump({
            'users': recs['user'].nunique(),
            'ndcg': ndcg
        }, fp)


if __name__ == '__main__':
    opts = docopt(__doc__)
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    logging.getLogger('numba').setLevel(logging.INFO)

    main(opts)