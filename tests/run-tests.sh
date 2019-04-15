#!/bin/bash

set -x

VECTORS=$@

if [ -z "$METHOD" ] ; then
    echo "METHOD needs to be set in the environment"
    exit 1;
fi

python cluster.py $VECTORS  --data-dir ../word-categorization/monolingual/en/ > ../../../results/$METHOD/cluster.csv
python outlier.py $VECTORS  --data-dir ../outlier-detection/monolingual/en/ > ../../../results/$METHOD/outlier.csv
python analogy.py $VECTORS  --data-dir ../word-analogy/monolingual/en/ > ../../../results/$METHOD/analogy.csv
python word-similarity-scores.py $VECTORS --data-dir ../word-similarity/monolingual/en/ > ../../../results/$METHOD/similarity.csv
