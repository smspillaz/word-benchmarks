#!/usr/bin/env python
"""Convert CSV table to LaTeX table."""
import argparse
import os
import pandas as pd
import numpy as np
import re
import sys

import markdown_table


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
    """Open CSV and print Markdown table."""
    parser = argparse.ArgumentParser("CSV2Markdown")
    parser.add_argument("csv")
    parser.add_argument("--caption")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    lines = [df.columns.values.tolist()] + df.reset_index(drop=True).values.tolist()

    header, lines = lines[:1], lines[1:]
    header = ["Model"] + [format_dataset(s) for s in header[0][1:]]
    # Sort based on word vector size
    sort_indices = np.argsort([int(re.sub(r".*?([0-9]+).*", r"\1", l[0])) for l in lines])
    lines = [lines[i] for i in sort_indices]

    print("### {}".format(args.caption))
    print("|{}|\n|{}|".format("|".join(header), "|".join([":--:" for x in header])))
    print("\n".join([
        "|{}|".format("|".join([str(x) for x in line]))
        for line in lines
    ]))


if __name__ == "__main__":
    main()
