import os
from argparse import ArgumentParser
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--counts", type=str)
    parser.add_argument("--train_h5", type=str)
    parser.add_argument("--test_h5", type=str)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--batch_start", type=int, default=0)
    parser.add_argument("--batch_end", type=int, default=None)
    parser.add_argument("--train_subset_pct", type=float, default=100)
    parser.add_argument("--random_seed", type=int, default=42)
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

    counts_df = pd.read_csv(args.counts, index_col="our_gene_name")
    gene_to_train_samples = get_samples(args.train_h5)
    gene_to_test_samples = get_samples(args.test_h5)
    assert len(gene_to_train_samples) == 400
    assert len(gene_to_test_samples) == 500  # also includes 100 unseen genes
    genes = sorted(gene_to_train_samples.keys())

    # Subset to genes within the specified batch
    if args.batch_end is None:
        args.batch_end = len(genes)
    my_genes = genes[args.batch_start : args.batch_end]
    counts_df = counts_df.loc[my_genes]

    # Create gene metadata file
    meta_df = counts_df[["Chr", "Coord"]]
    meta_df.to_csv(os.path.join(args.outdir, "gene_metadata.tsv"), index=True, sep="\t")

    rng = np.random.default_rng(seed=args.random_seed)

    # Create phenotype files
    for gene in my_genes:
        # Subsample train samples if necessary
        train_samples = list(gene_to_train_samples[gene])
        if args.train_subset_pct < 100:
            n_to_keep = int(len(train_samples) * args.train_subset_pct / 100)
            train_samples = rng.choice(train_samples, n_to_keep, replace=False)

        create_pheno_file(
            counts_df,
            gene,
            train_samples,
            os.path.join(args.outdir, f"{gene}.train.pheno"),
        )

        create_pheno_file(
            counts_df,
            gene,
            gene_to_test_samples[gene],
            os.path.join(args.outdir, f"{gene}.test.pheno"),
        )


if __name__ == "__main__":
    main()
