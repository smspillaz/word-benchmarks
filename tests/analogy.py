#!/usr/bin/env python
"""analogy.py

Test word vectors based on how well they are able to compute a given
analogy.

This is sort of tricky to do directly, but we basically simulate this
with vector artihmetic. Eg, King - Man + Woman = Queen.

The way that this is done is to do the arthimetic operations, then find
the most similar word in vector space. We then score based on how
similar the target word is to the result of the artihmetic operation.

For example, if we have Athens is to Greece as Baghdad is to Iraq, we
take Greece - Athens + Baghdad, then take the cosine distance to the target
word, "Iraq" and add that to our score. Higher scores are better.
"""

import argparse
import csv
import os
import sys

import numpy as np

from collections import namedtuple
from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity


Dataset = namedtuple("Dataset", "w1 w2 w3 target")

DATASETS = {
    "google-analogies.csv": Dataset(2, 3, 4, 5),
    "semeval.csv": Dataset(2, 3, 4, 5),
    "sat.csv": Dataset(2, 3, 4, 5),
    "msr.csv": Dataset(2, 3, 4, 5),
    "jair.csv": Dataset(1, 2, 3, 5)
}


def load_dataset(name, path):
    with open(path) as f:
        csv_data = csv.reader(f)

        _ = next(csv_data)
        metadata = DATASETS[name]

        for row in csv_data:
            first = row[metadata.w1].strip()
            second = row[metadata.w2].strip()
            third = row[metadata.w3].strip()
            target = row[metadata.target].strip()

            yield first, second, third, target


def score_dataset(model, dataset):
    """Get score for one dataset for a given KeyedVectors model."""
    similarities = [
        cosine_similarity(np.array([model[second] - model[first] + model[third]]),
                          np.array([model[target]]))[0]
        for first, second, third, target in dataset
        if (
            first in model.vocab and second in model.vocab and
            third in model.vocab and target in model.vocab
        )
    ]
    return np.mean(similarities) if similarities else 0


def score_model(model, datasets):
    """Get scores for all datasets."""

    for dataset_name in sorted(datasets.keys()):
        yield score_dataset(model, datasets[dataset_name])


def load_model(name):
    """Load a model."""
    return KeyedVectors.load(name).wv


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("""Word analogy score checker.""")
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