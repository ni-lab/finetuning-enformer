import os

import pandas as pd
from genomic_utils.variant import Variant
from tqdm import tqdm

FINEMAPPING_RESULTS_PATH = "/data/yosef3/scratch/ruchir/data/MAGE/MAGE.v1.0.data/QTL_results/eQTL_results/eQTL_finemapping_results/eQTL_finemapping.significantAssociations.MAGE.v1.0.txt"
VCF_PATH = "/data/yosef3/scratch/ruchir/data/MAGE/MAGE.v1.0.data/QTL_results/eQTL_results/eQTL_finemapping_results/hg19/significantVariants.hg38.vcf"


def main():
    results_df = pd.read_csv(
        FINEMAPPING_RESULTS_PATH,
        sep="\t",
        header=0,
        usecols=["variantChrom", "variantPosition", "variantRef", "variantAlt"],
    )

    variants = set()
    for _, row in tqdm(results_df.iterrows(), desc="Processing variants"):
        v = Variant(
            row["variantChrom"],
            row["variantPosition"],
            row["variantRef"],
            row["variantAlt"],
        )
        variants.add(v)
    print(f"Number of unique variants: {len(variants)}")

    # Sort the variants
    variants = sorted(variants)

    # Write the variants to a VCF file
    os.makedirs(os.path.dirname(VCF_PATH), exist_ok=True)
    with open(VCF_PATH, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(
            "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"])
            + "\n"
        )
        for v in tqdm(variants, desc="Writing VCF"):
            output_row = [f"chr{v.chrom}", v.pos, ".", v.ref, v.alt, ".", ".", "."]
            f.write("\t".join(map(str, output_row)) + "\n")


if __name__ == "__main__":
    main()
