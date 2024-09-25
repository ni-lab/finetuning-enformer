import sys
from collections import defaultdict

import h5py
import pandas as pd
from genomic_utils.variant import Variant
from tqdm import tqdm

sys.path += ["../..", "../../../vcf_utils"]
import evaluation_utils
import utils

INPUT_PATH = "GM12878_effect_sizes.csv"
OUTPUT_PATH = "GM12878_effect_sizes.trainingVariants.csv"

GEUVADIS_COUNTS_PATH = (
    "../../../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz"
)
TRAIN_H5_PATH = "../../../finetuning/data/h5_bins_384_chrom_split/train.h5"


def get_training_genes() -> set[str]:
    gene_to_class_map = evaluation_utils.get_gene_to_class_map()
    train_genes = {g for g, c in gene_to_class_map.items() if c != "unseen"}
    assert len(train_genes) == 400
    return train_genes


def get_samples_per_gene(h5_path: str) -> dict[str, set[str]]:
    gene_to_samples = defaultdict(set)
    with h5py.File(h5_path, "r") as f:
        genes = f["genes"][:].astype(str)
        samples = f["samples"][:].astype(str)
        for (g, s) in zip(genes, samples):
            gene_to_samples[g].add(s)
    return gene_to_samples


def get_common_variants(
    geno_mtx: pd.DataFrame, maf_threshold: float = 0.05
) -> set[Variant]:
    ac = geno_mtx.applymap(utils.convert_to_dosage).sum(axis=1)
    an = geno_mtx.applymap(utils.count_total_alleles).sum(axis=1)
    af = ac / an
    variants_f = af[(af >= maf_threshold) & (af <= 1 - maf_threshold)].index
    return set(variants_f)


def create_output_df(
    input_df: pd.DataFrame,
    variant_to_affected_gene: dict[Variant, set[str]],
):
    """Create a new DataFrame with the shared variants. Add the genes they affect and the
    GM12878_normalized_variant_effect."""
    output_data = [
        (v, g) for v, genes in variant_to_affected_gene.items() for g in genes
    ]
    output_df = pd.DataFrame(output_data, columns=["hg19_variant", "our_gene_name"])
    output_df = output_df.merge(input_df, on="hg19_variant")
    return output_df


def main():
    mpra_df = pd.read_csv(INPUT_PATH)
    if "hg19_variant" not in mpra_df:
        mpra_df["hg19_variant"] = [
            Variant(row["chr"], row["pos"], row["ref"], row["alt"])
            for _, row in mpra_df.iterrows()
        ]
    else:
        mpra_df["hg19_variant"] = mpra_df["hg19_variant"].apply(Variant.create_from_str)
    mpra_variants = set(mpra_df["hg19_variant"])

    # Create a map from variant to genes it affects in the training set
    training_genes = get_training_genes()
    training_samples_per_gene = get_samples_per_gene(TRAIN_H5_PATH)
    counts_df = pd.read_csv(GEUVADIS_COUNTS_PATH, index_col="our_gene_name")

    shared_variant_to_genes = defaultdict(set)
    for gene in tqdm(training_genes):
        geno_mtx = utils.get_genotype_matrix(
            counts_df.loc[gene, "Chr"], counts_df.loc[gene, "Coord"], 128 * 384
        )
        train_samples = sorted(training_samples_per_gene[gene])
        train_geno_mtx = geno_mtx[train_samples]

        common_variants = get_common_variants(train_geno_mtx)
        shared_variants = mpra_variants & common_variants
        for variant in shared_variants:
            shared_variant_to_genes[variant].add(gene)

    # Print out the number of variants that affect multiple genes
    n_affecting_multiple = sum(
        len(genes) > 1 for genes in shared_variant_to_genes.values()
    )
    print(
        f"{n_affecting_multiple}/{len(shared_variant_to_genes)} variants affect multiple genes"
    )

    # Create the output DataFrame
    output_df = create_output_df(mpra_df, shared_variant_to_genes)
    num_variants_per_gene = output_df["our_gene_name"].value_counts()
    print(
        "There are {} genes with >= 5 variants and {} genes with >= 10 variants".format(
            (num_variants_per_gene >= 5).sum(), (num_variants_per_gene >= 10).sum()
        )
    )

    output_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
