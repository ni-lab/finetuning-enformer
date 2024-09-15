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

INPUT_FINEMAPPING_PATH = "/data/yosef3/scratch/ruchir/data/MAGE/MAGE.v1.0.data/QTL_results/eQTL_results/eQTL_finemapping_results/hg19/eQTL_finemapping.significantAssociations.MAGE.v1.0.txt"
OUTPUT_PATH = (
    "eQTL_finemapping.significantAssociations.trainingVariants.MAGE.v1.0.hg19.txt"
)
GEUVADIS_COUNTS_PATH = (
    "../../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz"
)
TRAIN_H5_PATH = "../../finetuning/data/h5_bins_384_chrom_split/train.h5"


def get_training_genes() -> set[str]:
    gene_to_class_map = evaluation_utils.get_gene_to_class_map()
    train_genes = {g for g, c in gene_to_class_map.items() if c != "unseen"}
    assert len(train_genes) == 400
    return train_genes


def subset_to_training_genes(
    fmap_df: pd.DataFrame, counts_df: pd.DataFrame
) -> pd.DataFrame:
    # Check if geneSymbol is in our dataset
    fmap_df["name1"] = fmap_df["geneSymbol"].map(
        lambda n: n.lower() if n.lower() in counts_df.index else None
    )

    # Check if ensemblID is in our dataset
    counts_df["ensemblID"] = (
        counts_df["Gene_Symbol"].str.split(".").str[0]
    )  # remove version #
    fmap_df["ensemblID"] = (
        fmap_df["ensemblID"].str.split(".").str[0]
    )  # remove version #
    ensembl_to_name = {
        id_: name for id_, name in zip(counts_df["ensemblID"], counts_df.index)
    }
    fmap_df["name2"] = fmap_df["ensemblID"].map(ensembl_to_name)

    # Ensure that names match if they are not None
    mask = fmap_df["name1"].notnull() & fmap_df["name2"].notnull()
    assert fmap_df.loc[mask, "name1"].equals(fmap_df.loc[mask, "name2"])

    # Combine the two name columns
    fmap_df["our_gene_name"] = fmap_df["name1"].combine_first(fmap_df["name2"])
    fmap_df = fmap_df.drop(labels=["name1", "name2"], axis=1)

    return fmap_df[fmap_df["our_gene_name"].notnull()]


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
    af = ac / an
    af.index = af.index.map(str)
    return af


def subset_to_training_variants(
    fmap_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    context_size: int = 128 * 384,
    maf_threshold: float = 0.05,
) -> pd.DataFrame:
    # Get training samples
    gene_to_train_samples = get_samples_per_gene(TRAIN_H5_PATH)

    # Filter to variants in our training dataset with MAF >= 0.05
    train_fmap_dfs = []
    for g in tqdm(
        fmap_df["our_gene_name"].unique(), desc="Filtering to training variants"
    ):
        genotype_mtx = utils.get_genotype_matrix(
            counts_df.loc[g, "Chr"], counts_df.loc[g, "Coord"], context_size
        )

        # Compute MAF and filter variants by MAF
        train_samples = sorted(gene_to_train_samples[g])
        afs = compute_maf(genotype_mtx[train_samples])
        variants_f = afs[(afs >= maf_threshold) & (afs <= 1 - maf_threshold)].index

        # Filter to training variants
        gene_df = fmap_df[
            (fmap_df["our_gene_name"] == g) & fmap_df["hg19_variant"].isin(variants_f)
        ].copy()

        # Add MAF
        gene_df["MAF"] = gene_df["hg19_variant"].map(afs)

        # Add distance to TSS
        gene_df["hg19_pos"] = gene_df["hg19_variant"].apply(
            lambda v: Variant.create_from_str(v).pos
        )
        gene_df["TSS_dist"] = (gene_df["hg19_pos"] - counts_df.loc[g, "Coord"]).abs()
        train_fmap_dfs.append(gene_df)

    return pd.concat(train_fmap_dfs, axis=0, ignore_index=True)


def print_summary_stats(
    fmap_df: pd.DataFrame, title: str, gene_colname: str = "our_gene_name"
):
    print(title + "\n" + "-" * len(title))

    def _get_num_egenes(df: pd.DataFrame) -> int:
        return df[gene_colname].nunique()

    def _get_num_eqtls(df: pd.DataFrame) -> int:
        return df.shape[0]

    print(
        "All eQTLs: # eGenes = {}, # eQTLs = {}".format(
            _get_num_egenes(fmap_df), _get_num_eqtls(fmap_df)
        )
    )
    print(
        "Top eQTLs (PIP > 0.9): # eGenes = {}, # eQTLs = {}".format(
            _get_num_egenes(fmap_df[fmap_df["variantPIP"] > 0.9].copy()),
            _get_num_eqtls(fmap_df[fmap_df["variantPIP"] > 0.9].copy()),
        )
    )
    print("")


def main():
    # Load data
    train_genes = get_training_genes()
    counts_df = pd.read_csv(GEUVADIS_COUNTS_PATH, index_col="our_gene_name")
    counts_df = counts_df[counts_df.index.isin(train_genes)]
    fmap_df = pd.read_csv(INPUT_FINEMAPPING_PATH, sep="\t")

    print_summary_stats(fmap_df, "Stats before any filtering", gene_colname="ensemblID")

    # Subset to training genes
    fmap_df = subset_to_training_genes(fmap_df, counts_df)
    print_summary_stats(fmap_df, "Stats after subsetting to genes in our dataset")

    # Subset to training variants (train MAF >= 0.05 and within context)
    fmap_df = subset_to_training_variants(fmap_df, counts_df)
    print_summary_stats(fmap_df, "Stats after subsetting to training variants")

    # Save
    fmap_df.to_csv(OUTPUT_PATH, sep="\t", index=False)


if __name__ == "__main__":
    main()
