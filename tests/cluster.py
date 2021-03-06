#!/usr/bin/env python
"""cluster.py

Test word vectors based based on how well they cluster
related works into topics.

We have some data and cluster labels. We cluster all of the data
with N cluster assignments, then compare our clustering with the
clustering that we had from the data.

Scoring is based on mutual information score
"""

import argparse
import csv
import itertools
import json
import os
import re
import sys

import numpy as np

from collections import namedtuple
from gensim.models import KeyedVectors

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score

from tqdm.auto import tqdm

from util import make_vec, make_tokenizer


Dataset = namedtuple("Dataset", "label item")

DATASETS = {
    "ap.csv": Dataset(1, 2),
    "battig.csv": Dataset(1, 2),
    "bless.csv": Dataset(1, 2),
    "essli-2008.csv": Dataset(1, 2),
}


def load_dataset(name, path):
    with open(path) as f:
        csv_data = csv.reader(f)

        _ = next(csv_data)
        metadata = DATASETS[name]

        for row in csv_data:
            # Go through each row
            label = row[metadata.label]
            item = row[metadata.item]

            if item:
                yield label, item


def dataset_rows_to_clusters(rows):
    for i, (_, group) in enumerate(itertools.groupby(rows, key=lambda x: x[0])):
        yield i, [e[1] for e in group]


def vectorize_word(word, model):
    """Vectorize the word.

    First try to get the vectors directly, then if that fails, split
    it on _ and " " to get the subword vectors. If that fails for any
    part of the word, throw an exception.
    """
    if word in model.vocab:
        return model[word]

    subwords = re.split(r"[_\s]+", word)

    for subword in subwords:
        if not subword in model.vocab:
            raise KeyError

    return np.mean(np.stack([
        model[subword] for subword in subwords
    ]), axis=0)


def try_vectorize_word(word, model):
    try:
        return vectorize_word(word, model)
    except KeyError:
        return None


def yield_item_vectors(clusters, model, tokenizer):
    tfunc = tokenizer.tokenize if tokenizer is not None else lambda x: [x]
    for cluster in tqdm(clusters, desc="Processing clusters"):
        cluster_vectors = [
            make_vec(model, tfunc(item))
            for item in cluster if tokenizer is not None or item in model.vocab
        ]
        if cluster_vectors:
            yield np.array(cluster_vectors)


def yield_label_vectors(labels_clusters, model, tokenizer):
    for label, cluster in tqdm(labels_clusters, desc="Processing labels"):
        labels_vector = [label for item in cluster if tokenizer is not None or item in model.vocab]
        if labels_vector:
            yield np.array(labels_vector)


def score_dataset(model, dataset, tokenizer):
    """Get score for one dataset for a given KeyedVectors model."""
    clustering_data = np.stack(list(itertools.chain.from_iterable(yield_item_vectors([
        cluster for label, cluster in tqdm(dataset, desc="Processing Dataset")
    ], model, tokenizer))))
    true_clustering = np.array(list(itertools.chain.from_iterable(yield_label_vectors(dataset, model, tokenizer))))
    vector_clustering = KMeans(n_clusters=np.max(true_clustering) + 1).fit_predict(clustering_data)

    return normalized_mutual_info_score(true_clustering, vector_clustering, average_method='arithmetic')


def score_model(model, datasets, tokenizer):
    """Get scores for all datasets."""

    for dataset_name in tqdm(sorted(datasets.keys()), desc="Processing Datasets"):
        yield score_dataset(model, datasets[dataset_name], tokenizer)


def load_model(name):
    """Load a model."""
    return KeyedVectors.load(name).wv


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("""Topic clustering score checker.""")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--benchmarks", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--data-dir", type=str, help="Path to data")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer to use")
    args = parser.parse_args()

    models = {
        os.path.basename(n): load_model(n) for n in args.models
    }

    datasets = {
        n: list(dataset_rows_to_clusters(load_dataset(n, os.path.join(args.data_dir, n)))) for n in args.benchmarks
    }
    tokenizer = make_tokenizer(args.tokenizer)

    output = csv.writer(sys.stdout)
    output.writerow(['model'] + sorted(datasets.keys()))
    for model_name in tqdm(sorted(models.keys()), desc="Processing Models"):
        output.writerow([model_name] + list(score_model(models[model_name], datasets, tokenizer)))


if __name__ == "__main__":
    main()
