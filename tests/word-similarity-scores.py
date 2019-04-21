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
    print(path)
    with open(path) as f:
        csv_data = csv.reader(f)

        _ = next(csv_data)
        metadata = DATASETS[name]

        for row in csv_data:
            first = metadata.postproc(row[metadata.first])
            second = metadata.postproc(row[metadata.second])
            similarity = float(row[metadata.similarity]) / metadata.scale

            yield first, second, similarity


def score_dataset(model, dataset):
    """Get score for one dataset for a given KeyedVectors model."""
    similarities = [
        abs(model.similarity(first, second) - similarity)
        for first, second, similarity in dataset
        if first in model.vocab and second in model.vocab
    ]
    return np.mean(similarities)


def score_model(model, datasets):
    """Get scores for all datasets."""

    for dataset_name in sorted(datasets.keys()):
        yield score_dataset(model, datasets[dataset_name])


def load_model(name):
    """Load a model."""
    return KeyedVectors.load(name)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("""Word similarity score checker.""")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--benchmarks", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--data-dir", type=str, help="Path to data")
    args = parser.parse_args()

    datasets = {
        n: list(load_dataset(n, os.path.join(args.data_dir, n))) for n in args.benchmarks
    }
    models = {
        os.path.basename(n): load_model(n) for n in args.models
    }

    output = csv.writer(sys.stdout)
    output.writerow(['model'] + sorted(datasets.keys()))
    for model_name in sorted(models.keys()):
        output.writerow([model_name] + list(score_model(models[model_name], datasets)))


if __name__ == "__main__":
    main()