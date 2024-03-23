from argparse import ArgumentParser
from collections import defaultdict
from functools import partial

import h5py
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_path", type=str)
    # fmt: off
    parser.add_argument("--train_h5_path", type=str, default="../finetuning/data/h5_bins_384_chrom_split/train.h5")
    parser.add_argument("--val_h5_path", type=str, default="../finetuning/data/h5_bins_384_chrom_split/val.h5")
    parser.add_argument("--test_h5_path", type=str, default="../finetuning/data/h5_bins_384_chrom_split/test.h5")
    # fmt: on
    return parser.parse_args()


def get_samples(h5_split_path: str) -> dict[str, set]:
    """
    Returns a dictionary mapping genes to a set of samples for that gene found
    within the split.
    """
    gene_to_samples = defaultdict(set)
    with h5py.File(h5_split_path, "r") as f:
        genes = f["genes"][:].astype(str)
        samples = f["samples"][:].astype(str)
        for (gene, sample) in zip(genes, samples):
            gene_to_samples[gene].add(sample)
    return gene_to_samples


def determine_sample_split(
    gene: str,
    sample: str,
    gene_to_train_samples: dict[str, set],
    gene_to_val_samples: dict[str, set],
    gene_to_test_samples: dict[str, set],
) -> str:
    if sample in gene_to_train_samples[gene]:
        return "train"
    elif sample in gene_to_val_samples[gene]:
        return "val"
    elif sample in gene_to_test_samples[gene]:
        return "test"
    else:
        raise Exception(f"Sample {sample} was not found in any split for gene {gene}")


def main():
    args = parse_args()
    gene_to_train_samples = get_samples(args.train_h5_path)
    gene_to_val_samples = get_samples(args.val_h5_path)
    gene_to_test_samples = get_samples(args.test_h5_path)

    # Get a sorted list of all genes
    genes = (
        gene_to_train_samples.keys()
        | gene_to_val_samples.keys()
        | gene_to_test_samples.keys()
    )
    genes = sorted(list(genes))
    assert len(genes) == 500, f"Found {len(genes)} genes"

    # Get a sorted list of all samples
    train_samples = set.union(*gene_to_train_samples.values())
    val_samples = set.union(*gene_to_val_samples.values())
    test_samples = set.union(*gene_to_test_samples.values())
    samples = sorted(list(train_samples | val_samples | test_samples))
    assert len(samples) == 421, f"Found {len(samples)} samples"

    # Collate sample splits into a matrix. Only retain genes that don't have all samples in the
    # same split (i.e. remove unseen genes)
    determine_sample_split_fn = partial(
        determine_sample_split,
        gene_to_train_samples=gene_to_train_samples,
        gene_to_val_samples=gene_to_val_samples,
        gene_to_test_samples=gene_to_test_samples,
    )

    all_sample_splits = []
    genes_f = []

    for g in genes:
        gene_sample_splits = [determine_sample_split_fn(g, s) for s in samples]
        if len(set(gene_sample_splits)) == 1:
            continue
        all_sample_splits.append(gene_sample_splits)
        genes_f.append(g)

    sample_splits_df = pd.DataFrame(all_sample_splits, index=genes_f, columns=samples)
    sample_splits_df.to_csv(args.output_path)


if __name__ == "__main__":
    main()
