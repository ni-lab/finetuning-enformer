#!/bin/bash
# In this script, we scale the number of samples used to train a model to see how it affects
# performance.

# SCRIPT LOCATIONS
GCTA="/data/yosef3/scratch/ruchir/repos/fusion_twas/gcta_nr_robust"
PLINK="/data/yosef3/scratch/ruchir/tools/plink/plink"
GEMMA="/data/yosef3/scratch/ruchir/tools/gemma/gemma-0.98.5-linux-static-AMD64"

# ROWS IN THE MATRIX TO ANALYZE (FOR BATCHED RUNS)
BATCH_START=1
BATCH_END=400

# CONSTANTS
CONTEXT_SIZE=49152

MODELS="blup"

# PATHS
GEXP="../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz"
TRAIN_H5="../finetuning/data/h5_bins_384_chrom_split/train.h5"
TEST_H5="../finetuning/data/h5_bins_384_chrom_split/test.h5"
GENOTYPES="/data/yosef3/scratch/ruchir/data/geuvadis/genotypes/plink"

TMP_DIR="./tmp_rare_variant_analysis"
OUT_DIR="./WEIGHTS_rare_variant_analysis"

# --- BEGIN SCRIPT:
NR="${BATCH_START}_${BATCH_END}"

# Add symbolic link from output to the current directory in order to run BLUP and BSLMM
if [ -L output ]; then
	rm output
fi
ln -s ./ output

for maf in $(seq 0 0.01 0.09) $(seq 0.10 0.05 0.40) $(seq 0.41 0.01 0.49); do
	echo "Processing MAF ${maf}"

	# Create directories
	tmp_subdir="${TMP_DIR}/maf_${maf}/${NR}"
	out_subdir="${OUT_DIR}/maf_${maf}"
	mkdir -p ${tmp_subdir}
	mkdir -p ${out_subdir}

	# Create metadata file (one for each batch) and phenotype files (one for each gene)
	source ~/.bashrc
	conda activate sc
	python create_metadata_and_pheno_files.py \
		--counts $GEXP \
		--train_h5 $TRAIN_H5 \
		--test_h5 $TEST_H5 \
		--outdir "$tmp_subdir" \
		--batch_start $((BATCH_START - 1)) \
		--batch_end $BATCH_END

	# Loop through each gene in the batch
	conda activate r-env
	half_context_size=$((CONTEXT_SIZE / 2))

	cat "${tmp_subdir}/gene_metadata.tsv" | awk 'NR>1' | while read PARAM; do
		GNAME=`echo $PARAM | awk '{ print $1 }'`
		CHR=`echo $PARAM | awk '{ print $2 }'`
		TSS=`echo $PARAM | awk '{ print $3 }'`
		P0=$((TSS - half_context_size))
		P1=$((TSS + half_context_size - 1))

		echo "Processing gene ${GNAME} on chromosome ${CHR} from ${P0} to ${P1}"

		# Conditionally pass maf argument to plink command, as it errors out if maf is 0
		if (( $(echo "${maf} > 0" | bc -l) )); then
			MAF_ARG="--maf ${maf}"
		else
			MAF_ARG=""
		fi

		# Get the locus genotypes for all samples and set current gene expression as the phenotype
		${PLINK} \
			--bfile "${GENOTYPES}/GEUVADIS.chr${CHR}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes" \
			--pheno "${tmp_subdir}/${GNAME}.train.pheno" \
			--make-bed \
			--out "${tmp_subdir}/${GNAME}.train" \
			--keep "${tmp_subdir}/${GNAME}.train.pheno" \
			--chr ${CHR} \
			--from-bp ${P0} \
			--to-bp ${P1} \
			${MAF_ARG}  \
			--allow-no-sex \
			--snps-only \
			--keep-allele-order

		# Run FUSION.compute_weights.R
		Rscript /data/yosef3/scratch/ruchir/repos/fusion_twas/FUSION.compute_weights.R \
			--bfile "${tmp_subdir}/${GNAME}.train" \
			--tmp "${tmp_subdir}/${GNAME}.tmp" \
			--out "${out_subdir}/${GNAME}" \
			--verbose 1 \
			--save_hsq \
			--PATH_plink ${PLINK} \
			--PATH_gcta ${GCTA} \
			--PATH_gemma ${GEMMA} \
			--models blup \
			--crossval 0

		# Save weights
		Rscript save_weights.R \
			--RData "${out_subdir}/${GNAME}.wgt.RDat" \
			--out_prefix "${out_subdir}/${GNAME}"
	done
done
