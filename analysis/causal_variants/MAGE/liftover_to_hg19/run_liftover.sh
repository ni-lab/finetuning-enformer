INPUT_VCF="/data/yosef3/scratch/ruchir/data/MAGE/MAGE.v1.0.data/QTL_results/eQTL_results/eQTL_finemapping_results/hg19/significantVariants.hg38.vcf"
OUTPUT_VCF="/data/yosef3/scratch/ruchir/data/MAGE/MAGE.v1.0.data/QTL_results/eQTL_results/eQTL_finemapping_results/hg19/significantVariants.hg19.vcf"
REJECT_VCF="/data/yosef3/scratch/ruchir/data/MAGE/MAGE.v1.0.data/QTL_results/eQTL_results/eQTL_finemapping_results/hg19/liftoverRejectedVariants.vcf"

CHAIN_PATH="/data/yosef3/scratch/ruchir/data/genomes/liftOver/hg38ToHg19.over.chain.gz"
HG19_PATH="/data/yosef3/scratch/ruchir/data/genomes/hg19/hg19.fa"

java -jar /data/yosef3/scratch/ruchir/repos/picard/build/libs/picard.jar LiftoverVcf \
    --INPUT $INPUT_VCF \
    --OUTPUT $OUTPUT_VCF \
    --REJECT $REJECT_VCF \
    --CHAIN $CHAIN_PATH \
    --REFERENCE_SEQUENCE $HG19_PATH \
    --ALLOW_MISSING_FIELDS_IN_HEADER \
    --WRITE_ORIGINAL_POSITION \
    --WRITE_ORIGINAL_ALLELES
