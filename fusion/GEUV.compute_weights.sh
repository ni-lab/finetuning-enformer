#!/bin/bash
# SCRIPT LOCATIONS
GCTA="/data/yosef3/scratch/ruchir/repos/fusion_twas/gcta_nr_robust"
PLINK="/data/yosef3/scratch/ruchir/tools/plink/plink"
GEMMA="/data/yosef3/scratch/ruchir/tools/gemma/gemma-0.98.5-linux-static-AMD64"

# ROWS IN THE MATRIX TO ANALYZE (FOR BATCHED RUNS)
BATCH_START=1
BATCH_END=400

# CONSTANTS
CONTEXT_SIZE=49152
MAF=0.05

# PATHS
GEXP="../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz"
TRAIN_H5="../finetuning/data/h5_bins_384_chrom_split/train.h5"
TEST_H5="../finetuning/data/h5_bins_384_chrom_split/test.h5"
GENOTYPES="/data/yosef3/scratch/ruchir/data/geuvadis/genotypes/plink"

HSQ_DIR="./hsq"
TMP_DIR="./tmp"
OUT_DIR="./WEIGHTS"


# --- BEGIN SCRIPT:
NR="${BATCH_START}_${BATCH_END}"

mkdir -p ${HSQ_DIR}
mkdir -p ${TMP_DIR}/${NR}
mkdir -p ${OUT_DIR}

# Add symbolic link from output to the current directory in order to run BLUP and BSLMM
if [ -L output ]; then
	rm output
fi
ln -s ./ output

# Create metadata file (one for entire batch) and phenotype files (one for each gene)
source ~/.bashrc
conda activate sc
python create_metadata_and_pheno_files.py \
	--counts $GEXP \
	--train_h5 $TRAIN_H5 \
	--test_h5 $TEST_H5 \
	--outdir "$TMP_DIR/$NR" \
	--batch_start $((BATCH_START - 1)) \
	--batch_end $BATCH_END


# Loop through each gene in the batch
conda activate r-env
half_context_size=$((CONTEXT_SIZE / 2))

cat "${TMP_DIR}/${NR}/gene_metadata.tsv" | awk 'NR>1' | while read PARAM; do
	GNAME=`echo $PARAM | awk '{ print $1 }'`
	CHR=`echo $PARAM | awk '{ print $2 }'`
	TSS=`echo $PARAM | awk '{ print $3 }'`
	P0=$((TSS - half_context_size))
	P1=$((TSS + half_context_size - 1))

	echo "Processing gene ${GNAME} on chromosome ${CHR} from ${P0} to ${P1}"

	# Get the locus genotypes for all samples and set current gene expression as the phenotype
	${PLINK} \
		--bfile "${GENOTYPES}/GEUVADIS.chr${CHR}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes" \
		--pheno "${TMP_DIR}/${NR}/${GNAME}.train.pheno" \
		--make-bed \
		--out "${TMP_DIR}/${NR}/${GNAME}.train" \
		--keep "${TMP_DIR}/${NR}/${GNAME}.train.pheno" \
		--chr ${CHR} \
		--from-bp ${P0} \
		--to-bp ${P1} \
		--maf ${MAF} \
		--allow-no-sex \
		--snps-only \
		--keep-allele-order

	# Run FUSION.compute_weights.R
	Rscript /data/yosef3/scratch/ruchir/repos/fusion_twas/FUSION.compute_weights.R \
		--bfile "${TMP_DIR}/${NR}/${GNAME}.train" \
		--tmp "${TMP_DIR}/${NR}/${GNAME}.tmp" \
		--out "${OUT_DIR}/${GNAME}" \
		--verbose 1 \
		--save_hsq \
		--PATH_plink ${PLINK} \
		--PATH_gcta ${GCTA} \
		--PATH_gemma ${GEMMA} \
		--crossval 0
		--models lasso,top1,enet,blup,bslmm

	# Save weights
	Rscript save_weights.R \
		--RData "${OUT_DIR}/${GNAME}.wgt.RDat" \
		--out_prefix "${OUT_DIR}/${GNAME}"

	# Append heritability output to hsq file
	cat "${OUT_DIR}/${GNAME}.hsq" >> "${HSQ_DIR}/${NR}.hsq"
done
