#!/bin/bash

set -e

CONTEXT_SIZES=(10000 20000 30000 40000 49152 75000 100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000)

for context_size in ${CONTEXT_SIZES[@]}; do
    echo "Predicting for context size ${context_size}"
    python ../predict.py \
        --gene_metadata_path "tmp_${context_size}/1_400/gene_metadata.tsv" \
        --weight_dir "WEIGHTS_${context_size}/" \
        --temp_dir "tmp_${context_size}/1_400/" \
        --output_dir "preds/${context_size}" \
        --models top1 lasso enet blup \
        --context_size ${context_size} &
done

wait
