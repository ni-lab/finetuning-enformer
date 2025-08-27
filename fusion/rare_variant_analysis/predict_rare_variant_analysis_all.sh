#!/bin/bash

set -e

for maf in $(seq 0 0.01 0.09) $(seq 0.10 0.05 0.40) $(seq 0.41 0.01 0.49); do
    echo "Predicting for maf ${maf}"
    python predict.py \
        --gene_metadata_path tmp_rare_variant_analysis/maf_${maf}/1_400/gene_metadata.tsv \
        --weight_dir WEIGHTS_rare_variant_analysis/maf_${maf}/ \
        --temp_dir tmp_rare_variant_analysis/maf_${maf}/1_400/ \
        --output_dir preds_rare_variant_analysis/maf_${maf} \
        --maf ${maf} \
        --models blup &
done

wait