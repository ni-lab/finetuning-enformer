import os
from argparse import ArgumentParser, BooleanOptionalAction
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_path", type=str)
    parser.add_argument("split_type", type=str, choices=["random", "population"])
    parser.add_argument("--n_val", type=int, default=17)
    parser.add_argument("--n_test", type=int, default=77)
    parser.add_argument("--n_total_samples", type=int, default=421)
    parser.add_argument("--random_seed", type=int, default=42)
    # fmt: off
    parser.add_argument("--consensus_seq_dir", type=str, default="/data/yosef3/scratch/ruchir/data/basenji2_consensus_seqs")
    parser.add_argument("--counts_path", type=str, default="/data/yosef3/users/ruchir/finetuning-enformer/process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--finetuning_dataset_dir", type=str, default="../../finetuning/data/h5_bins_384_chrom_split")
    parser.add_argument("--metadata_path", type=str, default="/data/yosef3/users/ruchir/pgp_uq/data/E-GEUV-1.sdrf.txt")
    # fmt: on
    return parser.parse_args()


def load_genes_and_samples(args) -> tuple[list[str], list[str]]:
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    counts_df = counts_df[counts_df["EUR_eGene"]]
    assert counts_df.index.notnull().all()
    genes = counts_df.index.tolist()
    assert len(genes) == 3259

    samples_w_counts = set(
        c for c in counts_df.columns if c.startswith("HG") or c.startswith("NA")
    )
    gene_seq_dir = os.path.join(args.consensus_seq_dir, genes[0])
    samples_w_seq = set(
        f.split(".")[0] for f in os.listdir(gene_seq_dir) if f != "ref.fa"
    )
    samples = list(samples_w_counts & samples_w_seq)
    assert len(samples) == args.n_total_samples

    return (sorted(genes), sorted(samples))


def load_predefined_splits_from_finetuning_dataset(args) -> defaultdict[dict]:
    """
    Load the finetuning splits for random_split genes if we are creating a random split and
    yri_split genes if we are creating a population split. This allows a consistent
    performance comparison between the linear methods and the finetuned models.

    Returns a dictionary of dictionaries mapping genes to samples to split.
    """
    gene_class_path = os.path.join(args.finetuning_dataset_dir, "gene_class.csv")
    gene_class_df = pd.read_csv(gene_class_path, index_col=0)
    if args.split_type == "random":
        predefined_split_genes = set(
            gene_class_df[gene_class_df["class"] == "random_split"].index
        )
    if args.split_type == "population":
        predefined_split_genes = set(
            gene_class_df[gene_class_df["class"] == "yri_split"].index
        )
    assert len(predefined_split_genes) == 200

    splits = defaultdict(dict)
    for split in ["train", "val", "test"]:
        h5_path = os.path.join(args.finetuning_dataset_dir, f"{split}.h5")
        with h5py.File(h5_path, "r") as f:
            genes = f["genes"][:].astype(str)
            samples = f["samples"][:].astype(str)
            for g, s in zip(genes, samples):
                if g in predefined_split_genes:
                    splits[g][s] = split
    return splits


def get_sample_ancestries(metadata_path: str) -> dict[str, str]:
    df = pd.read_csv(metadata_path, sep="\t", header=0, index_col=0)
    return df["Characteristics[ancestry category]"].to_dict()


def create_splits_dict(
    train_samples: list[str], val_samples: list[str], test_samples: list[str]
) -> dict[str, str]:
    splits = {}
    for s in train_samples:
        splits[s] = "train"
    for s in val_samples:
        splits[s] = "val"
    for s in test_samples:
        splits[s] = "test"
    return splits


def create_random_split(
    samples: list[str], n_val: int, n_test: int, rng: np.random.Generator
) -> dict[str, str]:
    samples = rng.permutation(samples)
    test_samples = samples[:n_test]
    val_samples = samples[n_test : n_test + n_val]
    train_samples = samples[n_test + n_val :]
    return create_splits_dict(train_samples, val_samples, test_samples)


def create_population_split(
    samples: list[str],
    sample_to_ancestry: dict[str, str],
    n_val: int,
    rng: np.random.Generator,
):
    samples = np.asarray(samples)
    ancestries = np.array([sample_to_ancestry[s] for s in samples])

    dev_idxs = np.where(ancestries != "Yoruba")[0]
    test_idxs = np.where(ancestries == "Yoruba")[0]
    dev_samples = samples[dev_idxs]
    test_samples = samples[test_idxs]

    dev_samples = rng.permutation(dev_samples)
    val_samples = dev_samples[:n_val]
    train_samples = dev_samples[n_val:]
    return create_splits_dict(train_samples, val_samples, test_samples)


def main():
    args = parse_args()

    rng = np.random.default_rng(args.random_seed)
    genes, samples = load_genes_and_samples(args)
    predefined_splits = load_predefined_splits_from_finetuning_dataset(args)
    sample_to_ancestry = get_sample_ancestries(args.metadata_path)

    all_splits = defaultdict(dict)
    for g in genes:
        if g in predefined_splits:
            # If the gene is in the predefined splits, use those splits
            all_splits[g] = predefined_splits[g]
            continue

        # Otherwise, create random splits or population splits
        if args.split_type == "random":
            all_splits[g] = create_random_split(samples, args.n_val, args.n_test, rng)
        if args.split_type == "population":
            all_splits[g] = create_population_split(
                samples, sample_to_ancestry, args.n_val, rng
            )

    all_splits_arr = [[all_splits[g][s] for s in samples] for g in genes]
    splits_df = pd.DataFrame(all_splits_arr, index=genes, columns=samples)

    # Make sure the number of samples in each split is correct
    n_train_per_gene = splits_df.eq("train").sum(axis=1)
    n_val_per_gene = splits_df.eq("val").sum(axis=1)
    n_test_per_gene = splits_df.eq("test").sum(axis=1)
    assert n_train_per_gene.eq(args.n_total_samples - args.n_val - args.n_test).all()
    assert n_val_per_gene.eq(args.n_val).all()
    assert n_test_per_gene.eq(args.n_test).all()

    splits_df.to_csv(args.output_path)


if __name__ == "__main__":
    main()
