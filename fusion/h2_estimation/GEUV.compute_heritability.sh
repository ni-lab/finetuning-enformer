#!/bin/bash
set -e

# SCRIPT LOCATIONS
GCTA="/data/yosef3/scratch/ruchir/repos/fusion_twas/gcta_nr_robust"
PLINK="/data/yosef3/scratch/ruchir/tools/plink/plink"
GEMMA="/data/yosef3/scratch/ruchir/tools/gemma/gemma-0.98.5-linux-static-AMD64"

# CONSTANTS
CONTEXT_SIZE=49152
MAF=0.05

# STRATEGY
STRATEGY="test_samples_with_train_variants" # "all_samples", "test_samples", "test_samples_with_train_variants"

# PATHS
GENOTYPES="/data/yosef3/scratch/ruchir/data/geuvadis/genotypes/plink"
PHENO_DIR="pheno_files"
TMP_DIR="./tmp"
OUTPUT_PATH="heritability.${STRATEGY}.tsv"


# --- BEGIN SCRIPT:
mkdir -p ${TMP_DIR}

# Loop through each gene in the batch
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate r-env
half_context_size=$((CONTEXT_SIZE / 2))

cat "${PHENO_DIR}/gene_metadata.tsv" | awk 'NR>1' | while read PARAM; do
	GNAME=`echo $PARAM | awk '{ print $1 }'`
	CHR=`echo $PARAM | awk '{ print $2 }'`
	TSS=`echo $PARAM | awk '{ print $3 }'`
	P0=$((TSS - half_context_size))
	P1=$((TSS + half_context_size - 1))

	echo "Processing gene ${GNAME} on chromosome ${CHR} from ${P0} to ${P1}"

	if [ "${STRATEGY}" = "all_samples" ]; then
		${PLINK} \
			--bfile "${GENOTYPES}/GEUVADIS.chr${CHR}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes" \
			--pheno "${PHENO_DIR}/${GNAME}.all.pheno" \
			--keep "${PHENO_DIR}/${GNAME}.all.pheno" \
			--make-bed \
			--out "${TMP_DIR}/${GNAME}.plink_out" \
			--chr ${CHR} \
			--from-bp ${P0} \
			--to-bp ${P1} \
			--maf ${MAF} \
			--allow-no-sex \
			--snps-only \
			--keep-allele-order
	elif [ "${STRATEGY}" = "test_samples" ]; then
		${PLINK} \
			--bfile "${GENOTYPES}/GEUVADIS.chr${CHR}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes" \
			--pheno "${PHENO_DIR}/${GNAME}.test.pheno" \
			--keep "${PHENO_DIR}/${GNAME}.test.pheno" \
			--make-bed \
			--out "${TMP_DIR}/${GNAME}.plink_out" \
			--chr ${CHR} \
			--from-bp ${P0} \
			--to-bp ${P1} \
			--maf ${MAF} \
			--allow-no-sex \
			--snps-only \
			--keep-allele-order
	elif [ "${STRATEGY}" = "test_samples_with_train_variants" ]; then
		# Extract the variants from the training set (write-snplist)
		${PLINK} \
			--bfile "${GENOTYPES}/GEUVADIS.chr${CHR}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes" \
			--keep "${PHENO_DIR}/${GNAME}.train.pheno" \
			--write-snplist \
			--out "${TMP_DIR}/${GNAME}.train.snps" \
			--chr ${CHR} \
			--from-bp ${P0} \
			--to-bp ${P1} \
			--maf ${MAF} \
			--allow-no-sex \
			--snps-only \
			--keep-allele-order
		
		# Create plink file with test samples and training variants
		${PLINK} \
			--bfile "${GENOTYPES}/GEUVADIS.chr${CHR}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes" \
			--pheno "${PHENO_DIR}/${GNAME}.test.pheno" \
			--keep "${PHENO_DIR}/${GNAME}.test.pheno" \
			--extract "${TMP_DIR}/${GNAME}.train.snps.snplist" \
			--make-bed \
			--out "${TMP_DIR}/${GNAME}.plink_out" \
			--chr ${CHR} \
			--from-bp ${P0} \
			--to-bp ${P1} \
			--allow-no-sex \
			--snps-only \
			--keep-allele-order
	fi


	# Run FUSION.compute_weights.R
	Rscript /data/yosef3/scratch/ruchir/repos/fusion_twas/FUSION.compute_weights.R \
		--bfile "${TMP_DIR}/${GNAME}.plink_out" \
		--tmp "${TMP_DIR}/${GNAME}.tmp" \
		--out "${TMP_DIR}/${GNAME}" \
		--verbose 1 \
		--save_hsq \
		--PATH_plink ${PLINK} \
		--PATH_gcta ${GCTA} \
		--PATH_gemma ${GEMMA} \
		--crossval 0 \
		--models "top1"

	# Append heritability output to hsq file
	gene_hsq_file="${TMP_DIR}/${GNAME}.hsq"
	if [[ -f "${gene_hsq_file}" ]]; then
		cat "${gene_hsq_file}" >> ${OUTPUT_PATH}
	else
		echo "Error: Heritability file for ${GNAME} not found."
		echo "./tmp/${GNAME} 0.00 0.00 0.00" >> ${OUTPUT_PATH}
	fi
done
