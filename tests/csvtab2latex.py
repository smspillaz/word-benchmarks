#!/usr/bin/env python
"""Convert CSV table to LaTeX table."""
import argparse
import os
import pandas as pd
import numpy as np
import re
import sys


DATASET_NAMES = {
    "mc-30": "MC-30",
    "men": "MEN",
    "mturk-287": "MTurk-287",
    "mturk-771": "MTurk-771",
    "rg-65": "RG-65",
    "rw": "RW",
    "semeval17": "SemEval17",
    "simverb-3500": "SimVerb-3500",
    "verb-143": "Verb-143",
    "wordsim353-rel": "WordSim353-REL",
    "wordsim353-sim": "WordSim353-SIM",
    "yp-130": "YP-130",
    "8-8-8": "8-8-8",
    "wikisem500": "WikiSem-500",
    "wordsim-500": "WordSim-500",
    "google-analogies": "Google",
    "jair": "JAIR",
    "msr": "MSR",
    "sat": "SAT",
    "semeval": "SemEval17",
    "ap": "AP",
    "battig": "Battig",
    "bless": "BLESS",
    "essli-2008": "ESSLI-2008"
}


def format_dataset(dataset):
    dataset, _ = os.path.splitext(dataset)
    return DATASET_NAMES[dataset]


def main():
    """Open CSV and print LaTeX table."""
    parser = argparse.ArgumentParser("CSV2Latex")
    parser.add_argument("csv")
    parser.add_argument("--label")
    parser.add_argument("--caption")
    args = parser.parse_args()

    lines = pd.read_csv(args.csv).T.reset_index().values.tolist()

    header, lines = lines[:1], lines[1:]
    header = ["Dataset"] + [re.sub(r"(.*)?\.([0-9]+).*", r"\1 \2", s) for s in header[0][1:]]
    sort_indices = np.argsort([int(re.sub(r".*?([0-9]+).*", r"\1", h)) for h in header[1:]])
    header = header[:1] + [header[1:][i] for i in sort_indices]
    lines = [
        [format_dataset(line[0])] + [line[1:][i] for i in sort_indices]
        for line in lines 
    ]

    # Sort the lines based on the word vector size

    print(r"\begin{table}[]")
    print(r"\centering")
    print(r"\begin{tabular}{l|" + "l" * (len(header) - 1) + "}")
    print(r"\hline")
    print(" & ".join(header) + " \\\\")
    print(r"\hline")
    print(" \\\\ \n".join([
        " & ".join(line[:1] + ["{0:.3f}".format(float(l)) for l in line[1:]]) for line in lines
    ]))
    print(r"\end{tabular}")
    print(r"\caption{" + args.caption + "}")
    print(r"\label{tab:" + args.label + "}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
