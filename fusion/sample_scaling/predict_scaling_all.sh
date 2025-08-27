#!/bin/bash

set -e

# SUBSET_PCTS=(20 40 60 80 100)
SUBSET_PCTS=(40 60 80 100)
NUM_REPLICATES=3

for subset_pct in ${SUBSET_PCTS[@]}; do
	for ((replicate=1; replicate<=NUM_REPLICATES; replicate++)); do
        echo "Predicting for subset ${subset_pct} and replicate ${replicate}"
        python predict.py \
            --gene_metadata_path tmp_scaling/subset_${subset_pct}/replicate_${replicate}/1_400/gene_metadata.tsv \
            --weight_dir WEIGHTS_scaling/subset_${subset_pct}/replicate_${replicate}/ \
            --temp_dir tmp_scaling/subset_${subset_pct}/replicate_${replicate}/1_400/ \
            --output_dir preds_scaling/subset_${subset_pct}/replicate_${replicate} \
            --models blup &
    done
done

wait