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

from tqdm.auto import tqdm

from util import make_vec, make_tokenizer


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


def score_dataset(model, dataset, tokenizer):
    """Get score for one dataset for a given KeyedVectors model."""
    tfunc = tokenizer.tokenize if tokenizer else lambda x: [x]
    similarities = [
        cosine_similarity(np.array([make_vec(model, tfunc(second)) - make_vec(model, tfunc(first)) + make_vec(model, tfunc(third))]),
                          np.array([make_vec(model, tfunc(target))]))[0]
        for first, second, third, target in tqdm(dataset, desc="Processing dataset")
        if (
            tokenizer is not None or (
                first in model.vocab and second in model.vocab and
                third in model.vocab and target in model.vocab
            )
        )
    ]
    return np.mean(similarities) if similarities else 0


def score_model(model, datasets, tokenizer):
    """Get scores for all datasets."""

    for dataset_name in tqdm(sorted(datasets.keys()), desc="Processing datasets"):
        yield score_dataset(model, datasets[dataset_name], tokenizer)


def load_model(name):
    """Load a model."""
    return KeyedVectors.load(name).wv


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("""Word analogy score checker.""")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--benchmarks", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--data-dir", type=str, help="Path to data")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer to use.")
    args = parser.parse_args()

    datasets = {
        n: list(load_dataset(n, os.path.join(args.data_dir, n))) for n in args.benchmarks
    }
    models = {
        os.path.basename(n): load_model(n) for n in args.models
    }

    output = csv.writer(sys.stdout)
    output.writerow(['model'] + sorted(datasets.keys()))
    tokenizer = make_tokenizer(args.tokenizer)
    for model_name in tqdm(sorted(models.keys()), desc="Processing model"):
        output.writerow(
            [model_name] + list(score_model(models[model_name], datasets, tokenizer))
        )


if __name__ == "__main__":
    main()
