"""
We construct a dataset for finetuning. We consider the set of genes that are eGenes in the EUR
population and have their top eQTL within the context length. We then split these genes into 3 sets:
    - 50% for training and validation
    - 25% for testing (all samples)
    - 25% for testing (Yoruban samples)
"""

import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio import SeqIO
from joblib import Parallel, delayed
from scipy.stats import zscore


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("counts_path", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--total_n_genes", type=int, default=250)
    parser.add_argument("--context_length", type=int, default=128 * 100)
    parser.add_argument(
        "--consensus_seq_dir",
        type=str,
        default="/data/yosef3/scratch/ruchir/data/basenji2_consensus_seqs",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="/data/yosef3/users/ruchir/pgp_uq/data/E-GEUV-1.sdrf.txt",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def get_genes(args):
    """
    We randomly sample eGenes in the EUR population that have their top eQTL within the context length
    into 3 sets:
        - 50% for training and validation
        - 25% for testing (all samples)
        - 25% for testing (Yoruban samples, gene must be eGene in YRI population with top YRI eQTL within receptive field)
    """
    n_all_test_genes = int(0.25 * args.total_n_genes)
    n_yri_test_genes = int(0.25 * args.total_n_genes)
    n_dev_genes = args.total_n_genes - n_all_test_genes - n_yri_test_genes

    genes_df = pd.read_csv(args.counts_path, index_col=0)
    genes_df = genes_df[
        (genes_df["EUR_eGene"])
        & (genes_df["top_EUR_eqtl_distance"] < args.context_length // 2)
    ]
    genes_df = genes_df.reset_index().set_index("our_gene_name")
    assert genes_df.shape[0] >= args.total_n_genes

    yri_genes_df = genes_df[
        (genes_df["YRI_eGene"])
        & (genes_df["top_YRI_eqtl_distance"] < args.context_length // 2)
    ]
    assert yri_genes_df.shape[0] >= n_yri_test_genes

    rng = np.random.default_rng(args.random_seed)
    yri_test_genes = rng.choice(
        yri_genes_df.index, size=n_yri_test_genes, replace=False
    )
    remaining_genes = rng.permutation(genes_df.index.difference(yri_test_genes))

    return (
        remaining_genes[:n_dev_genes],
        remaining_genes[n_dev_genes : n_dev_genes + n_all_test_genes],
        yri_test_genes,
    )


def write_genes(dev_genes, all_test_genes, yri_test_genes, output_dir):
    output_path = os.path.join(output_dir, "genes.npz")
    np.savez(
        output_path,
        dev_genes=dev_genes,
        all_test_genes=all_test_genes,
        yri_test_genes=yri_test_genes,
    )


def get_seq_from_fasta(fasta_path: str, context_length: int) -> np.ndarray:
    record = SeqIO.read(fasta_path, "fasta")
    seq = str(record.seq).upper()
    bp_start = (len(seq) - context_length) // 2
    bp_end = bp_start + context_length
    seq = seq[bp_start:bp_end]
    return np.asarray([ord(c) for c in seq])


def load_data_for_gene(args, gene):
    """
    Return tuple of:
        ref_seq: (context_length,)
        sample_seqs: (n_samples, 2, context_length)
        samples: (n_samples,)
        Y: (n_samples,)
        Z: (n_samples,)
    """
    counts_df = pd.read_csv(args.counts_path, index_col=0)
    counts_df = counts_df.reset_index().set_index("our_gene_name")
    samples_with_counts = set(
        c for c in counts_df.columns if c.startswith("HG") or c.startswith("NA")
    )

    gene_seq_dir = os.path.join(args.consensus_seq_dir, gene)
    samples_with_gt = set(
        s.split(".")[0] for s in os.listdir(gene_seq_dir) if s != "ref.fa"
    )
    samples = np.asarray(list(sorted(samples_with_counts & samples_with_gt)))
    assert len(samples) == 421, f"{gene} has {len(samples)} samples, expected 421"

    Y = counts_df.loc[gene, samples].to_numpy().astype(np.float32)
    assert np.isnan(Y).sum() == 0
    Z = zscore(Y)

    ref_seq = get_seq_from_fasta(
        os.path.join(gene_seq_dir, "ref.fa"), args.context_length
    )

    sample_seqs = []
    for sample in samples:
        h1_fasta_path = os.path.join(gene_seq_dir, f"{sample}.1pIu.fa")
        h2_fasta_path = os.path.join(gene_seq_dir, f"{sample}.2pIu.fa")
        h1_seq = get_seq_from_fasta(h1_fasta_path, args.context_length)
        h2_seq = get_seq_from_fasta(h2_fasta_path, args.context_length)
        sample_seqs.append(np.stack([h1_seq, h2_seq], axis=0))
    sample_seqs = np.asarray(sample_seqs)

    return (ref_seq, sample_seqs, samples, Y, Z)


def get_sample_ancestries(metadata_path: str) -> dict[str, str]:
    df = pd.read_csv(metadata_path, sep="\t", header=0, index_col=0)
    return df["Characteristics[ancestry category]"].to_dict()


def add_sample_data(
    data: dict[str, list],
    seqs: np.ndarray,
    samples: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    gene: str,
):
    assert seqs.shape[0] == samples.size == Y.size == Z.size
    data["seqs"] += [seqs[i] for i in range(seqs.shape[0])]
    data["samples"] += samples.tolist()
    data["Y"] += Y.tolist()
    data["Z"] += Z.tolist()
    data["genes"] += [gene] * seqs.shape[0]


def split_dev(data: dict[str, np.ndarray]):
    n_samples = data["samples"].size
    n_train = int(n_samples * 0.9)

    permutation = np.random.permutation(n_samples)
    train_idxs = permutation[:n_train]
    val_idxs = permutation[n_train:]

    train_data = {k: v[train_idxs] for k, v in data.items()}
    val_data = {k: v[val_idxs] for k, v in data.items()}
    return (train_data, val_data)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dev_genes, all_test_genes, yri_test_genes = get_genes(args)
    write_genes(dev_genes, all_test_genes, yri_test_genes, args.output_dir)
    all_genes = np.concatenate((dev_genes, all_test_genes, yri_test_genes))
    print(f"# dev genes: {len(dev_genes)}")
    print(f"# all test genes: {len(all_test_genes)}")
    print(f"# YRI test genes: {len(yri_test_genes)}")

    all_results = Parallel(n_jobs=32, verbose=10)(
        delayed(load_data_for_gene)(args, g) for g in all_genes
    )

    ancestries = get_sample_ancestries(args.metadata_path)
    ref_data = defaultdict(list)
    sample_dev_data = defaultdict(list)
    sample_test_data = defaultdict(list)

    # Sort data into dev and test
    for gene, (ref_seq, sample_seqs, samples, Y, Z) in zip(all_genes, all_results):
        ref_data["genes"].append(gene)
        ref_data["ref_seqs"].append(ref_seq)

        if gene in dev_genes:
            add_sample_data(sample_dev_data, sample_seqs, samples, Y, Z, gene)
        elif gene in all_test_genes:
            add_sample_data(sample_test_data, sample_seqs, samples, Y, Z, gene)
        else:
            dev_idxs = [i for i, s in enumerate(samples) if ancestries[s] != "Yoruba"]
            test_idxs = [i for i, s in enumerate(samples) if ancestries[s] == "Yoruba"]
            add_sample_data(
                sample_dev_data,
                sample_seqs[dev_idxs],
                samples[dev_idxs],
                Y[dev_idxs],
                Z[dev_idxs],
                gene,
            )
            add_sample_data(
                sample_test_data,
                sample_seqs[test_idxs],
                samples[test_idxs],
                Y[test_idxs],
                Z[test_idxs],
                gene,
            )

    ref_data = {k: np.asarray(v) for k, v in ref_data.items()}
    sample_dev_data = {k: np.asarray(v) for k, v in sample_dev_data.items()}
    sample_test_data = {k: np.asarray(v) for k, v in sample_test_data.items()}
    sample_train_data, sample_val_data = split_dev(sample_dev_data)

    # Save data
    np.savez_compressed(os.path.join(args.output_dir, "ref.npz"), **ref_data)
    np.savez_compressed(
        os.path.join(args.output_dir, "sample_train.npz"), **sample_train_data
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "sample_val.npz"), **sample_val_data
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "sample_test.npz"), **sample_test_data
    )


if __name__ == "__main__":
    main()
