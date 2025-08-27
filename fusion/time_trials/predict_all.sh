#!/bin/bash

MODEL="bslmm"

python ../predict.py \
    --gene_metadata_path "tmp/1_400/gene_metadata.tsv" \
    --weight_dir "WEIGHTS/" \
    --temp_dir "tmp/1_400/" \
    --output_dir "preds" \
    --models ${MODEL} \
    --context_size 49152

echo "${MODEL}: ${SECONDS} seconds elapsed" >> "test_time_elapsed.txt"