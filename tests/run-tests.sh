#!/bin/bash

set -x

VECTORS=$@
TOKENIZER=${TOKENIZER:-none}

if [ -z "$METHOD" ] ; then
    echo "METHOD needs to be set in the environment"
    exit 1;
fi

mkdir -p ../../../tex-results/$METHOD/
mkdir -p ../../../md-results/$METHOD/
mkdir -p ../../../results/$METHOD/

python cluster.py $VECTORS  --data-dir ../word-categorization/monolingual/en/ > ../../../results/$METHOD/cluster.csv --tokenizer=$TOKENIZER
python csvtab2latex.py ../../../results/$METHOD/cluster.csv --label "cluster-$METHOD" --caption "Word Clustering ($METHOD) [1 is best]" > ../../../tex-results/$METHOD/cluster.tex
python csvtab2md.py ../../../results/$METHOD/cluster.csv --caption "Word Clustering ($METHOD) [1 is best]" > ../../../md-results/$METHOD/cluster.md
python outlier.py $VECTORS  --data-dir ../outlier-detection/monolingual/en/ > ../../../results/$METHOD/outlier.csv --tokenizer=$TOKENIZER
python csvtab2latex.py ../../../results/$METHOD/outlier.csv --label "outlier-$METHOD" --caption "Outlier Detection ($METHOD) [1 is best]" > ../../../tex-results/$METHOD/outlier.tex
python csvtab2md.py ../../../results/$METHOD/outlier.csv --caption "Outlier Detection ($METHOD) [1 is best]" > ../../../md-results/$METHOD/outlier.md
python analogy.py $VECTORS  --data-dir ../word-analogy/monolingual/en/ > ../../../results/$METHOD/analogy.csv --tokenizer=$TOKENIZER
python csvtab2latex.py ../../../results/$METHOD/analogy.csv --label "analogy-$METHOD" --caption "Word Analogy ($METHOD) [1 is best]" > ../../../tex-results/$METHOD/analogy.tex
python csvtab2md.py ../../../results/$METHOD/analogy.csv --caption "Word Analogy ($METHOD) [1 is best]" > ../../../md-results/$METHOD/analogy.md
python word-similarity-scores.py $VECTORS --data-dir ../word-similarity/monolingual/en/ > ../../../results/$METHOD/similarity.csv --tokenizer=$TOKENIZER
python csvtab2latex.py ../../../results/$METHOD/similarity.csv --label "similarity-$METHOD" --caption "Word Similarity, Mean Absolute Difference ($METHOD) [0 is best]" > ../../../tex-results/$METHOD/similarity.tex
python csvtab2md.py ../../../results/$METHOD/cluster.csv --caption "Word Similarity, Mean Absolute Difference ($METHOD) [0 is best]" > ../../../md-results/$METHOD/similarity.md
