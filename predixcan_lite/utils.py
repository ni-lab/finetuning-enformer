import os
import subprocess
from typing import Optional

import pandas as pd
from genomic_utils.variant import Variant
from snphwe import snphwe

GEUVADIS_GENOTYPES_DIR = "/data/yosef3/scratch/ruchir/data/geuvadis/genotypes"


def convert_to_dosage(genotype: str) -> int:
    if genotype == ".":
        return 0
    delim = "|" if "|" in genotype else "/"
    dosage = 0
    for allele in genotype.split(delim):
        assert allele in ["0", "1"]
        dosage += int(allele)
    return dosage


def count_total_alleles(genotype: str) -> int:
    if genotype == ".":
        return 0
    delim = "|" if "|" in genotype else "/"
    return len(genotype.split(delim))


def get_genotype_matrix(
    chrom: str, tss_pos: int, context_size: int, include_indels: bool = False
) -> pd.DataFrame:
    """
    Returns a genotype matrix for the given region. Dataframe is of shape [n_variants, n_samples]
    """
    chrom = chrom.replace("chr", "")
    start = tss_pos - context_size // 2
    start = max(start, 1)
    end = tss_pos + context_size - 1

    # Get variants in the region using tabix
    vcf_path = os.path.join(
        GEUVADIS_GENOTYPES_DIR,
        f"GEUVADIS.chr{chrom}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.vcf.bgz",
    )
    tabix_cmd = f"tabix -h {vcf_path} {chrom}:{start}-{end}"
    tabix_output = (
        subprocess.check_output(tabix_cmd, shell=True).decode("utf-8").split("\n")
    )

    samples = None
    variants, genotypes = [], []
    for line in tabix_output:
        if line.startswith("#CHROM"):
            samples = line.strip().split("\t")[9:]
        if not line or line.startswith("#"):
            continue
        row = line.strip().split("\t")
        variant = Variant(row[0], int(row[1]), row[3], row[4])

        # Skip variants that are not SNPs
        if not include_indels and (len(variant.ref) > 1 or len(variant.alt) > 1):
            continue

        variant_genotypes = [genotype.split(":")[0] for genotype in row[9:]]
        # Skip variants that do not have at least one alternate allele
        if not any(convert_to_dosage(gt) > 0 for gt in variant_genotypes):
            continue

        variants.append(variant)
        genotypes.append(variant_genotypes)

    genotype_matrix = pd.DataFrame(genotypes, index=variants, columns=samples)
    return genotype_matrix


def convert_to_dosage_matrix(genotype_matrix: pd.DataFrame) -> pd.DataFrame:
    return genotype_matrix.applymap(convert_to_dosage)


def filter_common(genotype_matrix: pd.DataFrame, maf: float = 0.05):
    AC = genotype_matrix.applymap(convert_to_dosage).sum(axis=1)
    AN = genotype_matrix.applymap(count_total_alleles).sum(axis=1)
    AF = AC / AN
    common_variants = AF[AF >= maf].index
    return genotype_matrix.loc[common_variants]


def filter_hwe(genotype_matrix: pd.DataFrame, min_pvalue: float = 0.05):
    AC = genotype_matrix.applymap(convert_to_dosage)
    AN = genotype_matrix.applymap(count_total_alleles)

    assert not (AN == 1).any().any()  # Nothing should be on a sex chromosome

    n_homo_refs = ((AC == 0) & (AN == 2)).sum(axis=1)
    n_homo_alts = ((AC == 2) & (AN == 2)).sum(axis=1)
    n_hets = ((AC == 1) & (AN == 2)).sum(axis=1)

    hwe_df = pd.DataFrame(
        {"n_homo_ref": n_homo_refs, "n_homo_alt": n_homo_alts, "n_het": n_hets}
    )

    hwe_df["hwe_p"] = [
        snphwe(n_het, n_homo_ref, n_homo_alt)
        for (n_homo_ref, n_homo_alt, n_het) in zip(
            hwe_df["n_homo_ref"], hwe_df["n_homo_alt"], hwe_df["n_het"]
        )
    ]

    hwe_variants = hwe_df[hwe_df >= min_pvalue].index
    return genotype_matrix.loc[hwe_variants]
