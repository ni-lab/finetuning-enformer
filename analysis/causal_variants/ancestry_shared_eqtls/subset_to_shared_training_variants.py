import gzip
import sys
from collections import defaultdict

import h5py
import pandas as pd
from genomic_utils.variant import Variant
from tqdm import tqdm

sys.path.append("../..")
import evaluation_utils

sys.path.append("../../../predixcan_lite")
import utils

EUR_EQTLS_PATH = "/data/yosef3/scratch/ruchir/data/geuvadis/analysis_results/EUR373.gene.cis.FDR5.all.rs137.txt.gz"
YRI_EQTLS_PATH = "/data/yosef3/scratch/ruchir/data/geuvadis/analysis_results/YRI89.gene.cis.FDR5.all.rs137.txt.gz"
DBSNP_TO_VARIANT_ID_PATH = "/data/yosef3/scratch/ruchir/data/geuvadis/genotypes/Phase1.Geuvadis_dbSnp137_idconvert.txt.gz"
GEUVADIS_VCF_WITHOUT_GENOTYPES_PATH = "/data/yosef3/scratch/ruchir/data/geuvadis/genotypes/ALL.phase1_release_v3.20101123.snps_indels_sv.sites.gdid.gdannot.v2.vcf.gz"
OUTPUT_PATH = "shared_training_eqtls.tsv"

GEUVADIS_COUNTS_PATH = (
    "../../../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz"
)
TRAIN_H5_PATH = "../../../finetuning/data/h5_bins_384_chrom_split/train.h5"


def get_training_genes() -> set[str]:
    gene_to_class_map = evaluation_utils.get_gene_to_class_map()
    train_genes = {g for g, c in gene_to_class_map.items() if c != "unseen"}
    assert len(train_genes) == 400
    return train_genes


def get_gene_id_to_name_map(counts_df: pd.DataFrame) -> dict[str, str]:
    id_to_name_map = {}
    for name, id_ in zip(counts_df.index, counts_df["TargetID"]):
        id_to_name_map[id_] = name
    return id_to_name_map


def create_dbsnp_to_variant_map(snp_ids: set[str]) -> dict[str, Variant]:
    # 1. Create a map from SNP_ID to VARIANT_ID
    # -----------------------------------------
    df = pd.read_csv(
        DBSNP_TO_VARIANT_ID_PATH, sep="\t", header=0, names=["SNP_ID", "VARIANT_ID"]
    )
    df = df[df["SNP_ID"].isin(snp_ids)]
    snp_to_variant_id_map = {}
    for snp_id, variant_id in zip(df["SNP_ID"], df["VARIANT_ID"]):
        if snp_id in snp_to_variant_id_map:
            # indels are sometimes mapped to multiple IDs. This is not a problem
            # because we don't deal with indels in the subsequent analysis.
            assert snp_to_variant_id_map[snp_id].startswith("indel")
        snp_to_variant_id_map[snp_id] = variant_id

    # 2. Create a map from VARIANT_ID to Variant
    # ------------------------------------------
    variant_ids = set(snp_to_variant_id_map.values())
    id_to_variant_map = {}
    with gzip.open(GEUVADIS_VCF_WITHOUT_GENOTYPES_PATH, "rt") as f:
        for line in tqdm(f, desc="Mapping variant IDs to variants"):
            if line.startswith("#"):
                continue
            row = line.strip().split("\t", maxsplit=5)
            id_ = row[2]
            if id_ not in variant_ids:
                continue
            assert id_ not in id_to_variant_map
            id_to_variant_map[id_] = Variant(row[0], row[1], row[3], row[4])

    # 3. Create a composite map from SNP_ID to Variant
    # -----------------------------------------------
    composite_map = {}
    for snp_id in snp_to_variant_id_map:
        variant_id = snp_to_variant_id_map[snp_id]
        composite_map[snp_id] = id_to_variant_map[variant_id]
    return composite_map


def get_samples_per_gene(h5_path: str) -> dict[str, set[str]]:
    gene_to_samples = defaultdict(set)
    with h5py.File(h5_path, "r") as f:
        genes = f["genes"][:].astype(str)
        samples = f["samples"][:].astype(str)
        for (g, s) in zip(genes, samples):
            gene_to_samples[g].add(s)
    return gene_to_samples


def compute_maf(genotype_mtx: pd.DataFrame) -> pd.Series:
    ac = genotype_mtx.applymap(utils.convert_to_dosage).sum(axis=1)
    an = genotype_mtx.applymap(utils.count_total_alleles).sum(axis=1)
    return ac / an


def subset_to_training_variants(
    shared_eqtls_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    context_size: int = 128 * 384,
    maf_threshold: float = 0.05,
) -> pd.DataFrame:
    # Get training samples
    gene_to_train_samples = get_samples_per_gene(TRAIN_H5_PATH)

    # Filter to variants in our training dataset with MAF >= 0.05
    per_gene_dfs = []
    for g in tqdm(
        shared_eqtls_df["our_gene_name"].unique(), desc="Filtering to training variants"
    ):
        genotype_mtx = utils.get_genotype_matrix(
            counts_df.loc[g, "Chr"], counts_df.loc[g, "Coord"], context_size
        )

        # Compute MAF and filter variants by MAF
        train_samples = sorted(gene_to_train_samples[g])
        afs = compute_maf(genotype_mtx[train_samples])
        variants_f = afs[(afs >= maf_threshold) & (afs <= 1 - maf_threshold)].index

        # Filter to training variants
        gene_df = shared_eqtls_df[
            (shared_eqtls_df["our_gene_name"] == g)
            & (shared_eqtls_df["hg19_variant"].isin(variants_f))
        ].copy()

        # Add MAF
        gene_df["MAF"] = gene_df["hg19_variant"].map(afs)

        # Add distance to TSS
        gene_df["hg19_pos"] = gene_df["hg19_variant"].apply(lambda v: v.pos)
        gene_df["TSS_dist"] = (gene_df["hg19_pos"] - counts_df.loc[g, "Coord"]).abs()
        per_gene_dfs.append(gene_df)

    return pd.concat(per_gene_dfs, axis=0, ignore_index=True)


def print_summary_stats(
    eqtls_df: pd.DataFrame, title: str, gene_colname: str = "our_gene_name"
):
    print(title + "\n" + "-" * len(title))
    print(
        "# eGenes = {}, # eQTLs = {}".format(
            eqtls_df[gene_colname].nunique(), eqtls_df.shape[0]
        )
    )
    print("")


def main():
    # Load EUR and YRI eQTLs
    eur_eqtls_df = pd.read_csv(EUR_EQTLS_PATH, sep="\t")
    yri_eqtls_df = pd.read_csv(YRI_EQTLS_PATH, sep="\t")

    # Identify variants that are identified as eQTLs in both the EUR and YRI populations.
    shared_eqtls_df = eur_eqtls_df.merge(
        yri_eqtls_df, on=["SNP_ID", "GENE_ID"], suffixes=("_EUR", "_YRI")
    )
    print_summary_stats(
        shared_eqtls_df, "Stats for all shared eQTLs", gene_colname="GENE_ID"
    )

    # Add common gene name
    counts_df = pd.read_csv(GEUVADIS_COUNTS_PATH, index_col="our_gene_name")
    gene_id_to_name_map = get_gene_id_to_name_map(counts_df)
    shared_eqtls_df["our_gene_name"] = shared_eqtls_df["GENE_ID"].map(
        gene_id_to_name_map
    )

    # Subset to training genes
    train_genes = get_training_genes()
    shared_eqtls_df = shared_eqtls_df[
        shared_eqtls_df["our_gene_name"].isin(train_genes)
    ]
    print_summary_stats(shared_eqtls_df, "Stats after filtering to training genes")

    # Add Variant information
    snp_id_to_variant_map = create_dbsnp_to_variant_map(set(shared_eqtls_df["SNP_ID"]))
    shared_eqtls_df["hg19_variant"] = shared_eqtls_df["SNP_ID"].map(
        snp_id_to_variant_map
    )

    # Filter to variants in our training dataset with MAF >= 0.05
    shared_eqtls_df = subset_to_training_variants(shared_eqtls_df, counts_df)
    print_summary_stats(shared_eqtls_df, "Stats after subsetting to training variants")

    # Save
    shared_eqtls_df.to_csv(OUTPUT_PATH, sep="\t", index=False)


if __name__ == "__main__":
    main()
