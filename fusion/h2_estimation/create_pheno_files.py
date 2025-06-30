import os
from argparse import ArgumentParser
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--counts_path", type=str, default="../../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--train_h5", type=str, default="../../finetuning/data/h5_bins_384_chrom_split/train.h5")
    parser.add_argument("--val_h5", type=str, default="../../finetuning/data/h5_bins_384_chrom_split/val.h5")
    parser.add_argument("--test_h5", type=str, default="../../finetuning/data/h5_bins_384_chrom_split/test.h5")
    parser.add_argument("--outdir", type=str, default="pheno_files")
    return parser.parse_args()


def get_samples(h5_path: str) -> dict[str, set]:
    gene_to_samples = defaultdict(set)
    with h5py.File(h5_path, "r") as f:
        genes = f["genes"][:].astype(str)
        samples = f["samples"][:].astype(str)
        for (gene, sample) in zip(genes, samples):
            gene_to_samples[gene].add(sample)
    return gene_to_samples


def create_pheno_file(
    counts_df: pd.DataFrame, gene: str, samples: list[str], output_path: str
):
    samples = sorted(samples)
    exprs = counts_df.loc[gene, samples].values
    output_df = pd.DataFrame({"sample1": samples, "sample2": samples, "expr": exprs})
    output_df.to_csv(output_path, index=False, header=False, sep=" ")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    gene_to_train_samples = get_samples(args.train_h5)
    gene_to_val_samples = get_samples(args.val_h5)
    gene_to_test_samples = get_samples(args.test_h5)
    assert len(gene_to_train_samples) == 400
    assert len(gene_to_val_samples) == 400
    assert len(gene_to_test_samples) == 500  # also includes 100 unseen genes
    train_genes = sorted(gene_to_train_samples.keys())

    # Subset to train genes
    counts_df = counts_df.loc[train_genes]

    # Create gene metadata file
    meta_df = counts_df[["Chr", "Coord"]]
    meta_df.to_csv(os.path.join(args.outdir, "gene_metadata.tsv"), index=True, sep="\t")

    # Create phenotype files
    for g in train_genes:
        # Create phenotype files for train/val/test splits
        train_samples = list(gene_to_train_samples[g])
        val_samples = list(gene_to_val_samples[g])
        test_samples = list(gene_to_test_samples[g])

        create_pheno_file(counts_df, g, train_samples, os.path.join(args.outdir, f"{g}.train.pheno"))
        create_pheno_file(counts_df, g, val_samples, os.path.join(args.outdir, f"{g}.val.pheno"))
        create_pheno_file(counts_df, g, test_samples, os.path.join(args.outdir, f"{g}.test.pheno"))

        # Create phenotype file for all samples together
        all_samples = train_samples + val_samples + test_samples
        create_pheno_file(counts_df, g, all_samples, os.path.join(args.outdir, f"{g}.all.pheno"))


if __name__ == "__main__":
    main()
