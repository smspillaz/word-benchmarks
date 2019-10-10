#!/usr/bin/env python
"""word-similarity-scores.py

Test word vectors and collect word similarity metrics from human
evaluations, measuring how close we were in absolute distance
between human word similarity and vector space word similarity
for a given model."""

import argparse
import csv
import os
import sys

import numpy as np

from collections import namedtuple
from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity

from tqdm.auto import tqdm

from util import make_vec, make_tokenizer


Dataset = namedtuple("Dataset", "first second similarity scale postproc")

DATASETS = {
    "mc-30.csv": Dataset(1, 2, 3 ,4, lambda x: x),
    "men.csv": Dataset(1, 2, 3, 50, lambda x: x[:-2]),
    "mturk-287.csv": Dataset(1, 2, 3, 5, lambda x: x),
    "mturk-771.csv": Dataset(1, 2, 3, 5, lambda x: x),
    "rg-65.csv": Dataset(1, 2, 3, 4, lambda x: x),
    "rw.csv": Dataset(1, 2, 3, 10, lambda x: x),
    "semeval17.csv": Dataset(1, 2, 3, 4, lambda x: x),
    "simverb-3500.csv": Dataset(2, 3, 1, 4, lambda x: x),
    "verb-143.csv": Dataset(1, 2, 3, 4, lambda x: x),
    "wordsim353-rel.csv": Dataset(1, 2, 3, 10, lambda x: x),
    "wordsim353-sim.csv": Dataset(1, 2, 3, 10, lambda x: x),
    "yp-130.csv": Dataset(1, 2, 3, 4, lambda x: x),
}


def load_dataset(name, path):
    with open(path) as f:
        csv_data = csv.reader(f)

        _ = next(csv_data)
        metadata = DATASETS[name]

        for row in csv_data:
            first = metadata.postproc(row[metadata.first])
            second = metadata.postproc(row[metadata.second])
            similarity = float(row[metadata.similarity]) / metadata.scale

            yield first, second, similarity


def score_dataset(model, dataset, tokenizer):
    """Get score for one dataset for a given KeyedVectors model."""
    tfunc = tokenizer.tokenize if tokenizer is not None else lambda x: [x]
    similarities = [
        abs(
            cosine_similarity(np.array([make_vec(model, tfunc(first))]),
                              np.array([make_vec(model, tfunc(second))]))[0] -
            similarity
        )
        for first, second, similarity in tqdm(dataset, "Processing Dataset")
        if tokenizer is not None or (first in model.vocab and second in model.vocab)
    ]
    return np.mean(similarities)


def score_model(model, datasets, tokenizer):
    """Get scores for all datasets."""

    for dataset_name in tqdm(sorted(datasets.keys()), desc="Processing Datasets"):
        yield score_dataset(model, datasets[dataset_name], tokenizer)


def load_model(name):
    """Load a model."""
    return KeyedVectors.load(name).wv


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("""Word similarity score checker.""")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--benchmarks", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--data-dir", type=str, help="Path to data")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer to use")
    args = parser.parse_args()

    datasets = {
        n: list(load_dataset(n, os.path.join(args.data_dir, n))) for n in args.benchmarks
    }
    models = {
        os.path.basename(n): load_model(n) for n in args.models
    }
    tokenizer = make_tokenizer(args.tokenizer)

    output = csv.writer(sys.stdout)
    output.writerow(['model'] + sorted(datasets.keys()))
    for model_name in sorted(models.keys()):
        output.writerow([model_name] + list(score_model(models[model_name], datasets, tokenizer)))


if __name__ == "__main__":
    main()
