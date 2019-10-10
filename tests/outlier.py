#!/usr/bin/env python
"""outlier.py

Test word vectors based on how well they can detect outliers.

Outlier detection is based on the compactness score of
each word, c(w) =  1/(n(n - 1)) * \sum_{w_i \in [W + w]} * \sum_{w_j \in [W + w], w_j != w_i} sim(w_i, w_j).
We then get the compactness score of each word, including the known outlier
rank them all by their compactness score (higher has more similarity). The
intuitive understanding of this is that we are taking the union of the outlier
and the cluster, then for each term in that cluster, computing the accumulated
intra-cluster similarity score. In principle, the outlier should have
the lowest score.

The scoring for outlier detection is based on (rank / n), eg, we get
the highest score if the outlier word has the lowest rank.
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

from collections import namedtuple
from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity

from tqdm.auto import tqdm

from util import make_vec, make_tokenizer


Dataset = namedtuple("Dataset", "outliers cluster")

DATASETS = {
    "8-8-8.csv": Dataset(2, 3),
    "wikisem500.csv": Dataset(2, 3),
}


def filter_invalid(wordset):
    return [x for x in wordset if len(x)]


def load_dataset(name, path):
    with open(path) as f:
        csv_data = csv.reader(f)

        _ = next(csv_data)
        metadata = DATASETS[name]

        for row in csv_data:
            # Work around invalid action.
            #
            # The string of replace's is a poor-mans hack to work around the
            # fact that re won't let you do a negative lookbehind on a backslash
            # even if you escape it.
            outliers = json.loads(re.sub("(?<=[\[\s,])'|'(?=[\]\s,])", "\"", row[metadata.outliers].strip()))
            cluster = json.loads(re.sub("(?<=[\[\s,])'|'(?=[\]\s,])", "\"", row[metadata.cluster].strip()))

            yield filter_invalid(outliers), filter_invalid(cluster)


def vectorize_word(word, model, tokenizer):
    """Vectorize the word.

    First try to get the vectors directly, then if that fails, split
    it on _ and " " to get the subword vectors. If that fails for any
    part of the word, throw an exception.
    """
    tfunc = tokenizer.tokenize if tokenizer else lambda x: [x]
    if tokenizer:
        return make_vec(model, tfunc(word))

    if word in model.vocab:
        return model[word]

    subwords = re.split(r"[_\s]+", word)

    for subword in subwords:
        if not subword in model.vocab:
            raise KeyError

    return np.mean(np.stack([
        model[subword] for subword in subwords
    ]), axis=0)


def process_cluster(words, model, tokenizer):
    for word in words:
        try:
            yield vectorize_word(word, model, tokenizer)
        except KeyError:
            pass


def simple_cosine_similarity(left, right):
    return cosine_similarity(np.array([left]), np.array([right]))[0]


def score_outliers(model, vectorized_outliers, vectorized_cluster):
    # Set with the outlier times set without it
    k = (len(vectorized_cluster) + 1) * (len(vectorized_cluster))

    # For each outlier in our set of ourliers
    for outlier in vectorized_outliers:
        cluster_with_outlier = np.concatenate([vectorized_cluster, np.array([outlier])])
        # Per-element scores (outlier is last in the list)
        p_scores = np.array([
            np.sum([
                simple_cosine_similarity(left_element, right_element)
                for j, right_element in enumerate(cluster_with_outlier)
                if i != j
            ])
            for i, left_element in enumerate(cluster_with_outlier)
        ]) / k

        # Now get the ranking of our outlier on the p-scores.
        yield list(reversed(np.argsort(p_scores)))[-1]


def score_cluster(model, outliers, cluster, tokenizer):
    vectorized_outliers = np.array(list(process_cluster(outliers, model, tokenizer)))
    vectorized_cluster = np.array(list(process_cluster(cluster, model, tokenizer)))

    if not len(vectorized_outliers) or not len(vectorized_cluster):
        return None

    positions = np.array(list(score_outliers(model,
                                             vectorized_outliers,
                                             vectorized_cluster)))
    return np.mean(positions) / len(vectorized_cluster)




def score_dataset(model, dataset, tokenizer):
    """Get score for one dataset for a given KeyedVectors model."""
    cluster_scores = list(filter(lambda x: x is not None, [
        score_cluster(model, outliers, cluster, tokenizer)
        for outliers, cluster in tqdm(dataset, desc="Processing dataset")
    ]))
    return np.mean(cluster_scores) if cluster_scores else 0


def score_model(model, datasets, tokenizer):
    """Get scores for all datasets."""

    for dataset_name in tqdm(sorted(datasets.keys()), desc="Processing Datasets"):
        yield score_dataset(model, datasets[dataset_name], tokenizer)


def load_model(name):
    """Load a model."""
    return KeyedVectors.load(name).wv


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("""Outlier detection score checker.""")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--benchmarks", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--data-dir", type=str, help="Path to data")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer to use")
    args = parser.parse_args()

    models = {
        os.path.basename(n): load_model(n) for n in args.models
    }

    datasets = {
        n: list(load_dataset(n, os.path.join(args.data_dir, n))) for n in args.benchmarks
    }

    output = csv.writer(sys.stdout)
    output.writerow(['model'] + sorted(datasets.keys()))
    tokenizer = make_tokenizer(args.tokenizer)
    for model_name in tqdm(sorted(models.keys()), desc="Processing Models"):
        output.writerow([model_name] + list(score_model(models[model_name], datasets, tokenizer)))


if __name__ == "__main__":
    main()
