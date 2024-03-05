"""
We construct a dataset for finetuning. The dataset consists of 3 classes of genes:
    - random_split genes: samples (EUR + YRI) randomly split between dev and test sets
    - yri_split_genes: EUR in dev, YRI in test
    - unseeen_genes: all samples in test
"""

import os
from argparse import ArgumentParser

import h5py
import numpy as np
import pandas as pd
from Bio import SeqIO
from enformer_pytorch import str_to_one_hot
from joblib import Parallel, delayed
from scipy.stats import percentileofscore, zscore
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--context_length", type=int, default=128 * 384)
    parser.add_argument("--n_random_split", type=int, default=200)
    parser.add_argument("--n_yri_split", type=int, default=200)
    parser.add_argument("--n_unseen", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    # fmt: off
    parser.add_argument("--counts_path", type=str, default="/data/yosef3/users/ruchir/finetuning-enformer/process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--consensus_seq_dir", type=str, default="/data/yosef3/scratch/ruchir/data/basenji2_consensus_seqs")
    parser.add_argument("--metadata_path", type=str, default="/data/yosef3/users/ruchir/pgp_uq/data/E-GEUV-1.sdrf.txt")
    # fmt: on
    return parser.parse_args()


def split_chromosomes_by_class(args, counts_df) -> dict[str, set]:
    counts_by_chrom = counts_df["Chr"].value_counts().to_dict()

    # Apportion chromosomes to each class in proportion to the number of genes
    # from each class we wish to have
    n_my_genes = args.n_random_split + args.n_yri_split + args.n_unseen
    targets = {
        "random_split": int(args.n_random_split / n_my_genes * counts_df.shape[0]),
        "yri_split": int(args.n_yri_split / n_my_genes * counts_df.shape[0]),
        "unseen": int(args.n_unseen / n_my_genes * counts_df.shape[0]),
    }

    chrom_by_class = {c: set() for c in targets}
    running_totals = {c: 0 for c in targets}

    for chrom in sorted(counts_by_chrom, key=counts_by_chrom.get, reverse=True):
        for clazz, target in targets.items():
            if (running_totals[clazz] + counts_by_chrom[chrom] <= target) or (
                clazz == "unseen"
            ):
                chrom_by_class[clazz].add(chrom)
                running_totals[clazz] += counts_by_chrom[chrom]
                break

    return chrom_by_class


def get_genes(args):
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    counts_df = counts_df[counts_df["EUR_eGene"]].copy()
    assert counts_df.index.notnull().all()
    chrom_by_class = split_chromosomes_by_class(args, counts_df)

    rng = np.random.default_rng(args.random_seed)

    random_split_genes = rng.choice(
        counts_df[counts_df["Chr"].isin(chrom_by_class["random_split"])].index,
        size=args.n_random_split,
        replace=False,
    )
    yri_split_genes = rng.choice(
        counts_df[counts_df["Chr"].isin(chrom_by_class["yri_split"])].index,
        size=args.n_yri_split,
        replace=False,
    )
    unseen_genes = rng.choice(
        counts_df[counts_df["Chr"].isin(chrom_by_class["unseen"])].index,
        size=args.n_unseen,
        replace=False,
    )

    class_to_gene = {
        "random_split": random_split_genes,
        "yri_split": yri_split_genes,
        "unseen": unseen_genes,
    }

    return {g: c for c, genes in class_to_gene.items() for g in genes}


def publish_gene_class_csv(output_dir, gene_to_class):
    gene_class_df = pd.DataFrame(list(gene_to_class.items()), columns=["gene", "class"])
    gene_class_df.to_csv(os.path.join(output_dir, "gene_class.csv"), index=False)


def get_seq_from_fasta(fasta_path: str, context_length: int) -> np.ndarray:
    record = SeqIO.read(fasta_path, "fasta")
    seq = str(record.seq).upper()
    bp_start = (len(seq) - context_length) // 2
    bp_end = bp_start + context_length
    seq = seq[bp_start:bp_end]
    return str_to_one_hot(seq).numpy().astype(np.float16)


def get_sample_ancestries(metadata_path: str) -> dict[str, str]:
    df = pd.read_csv(metadata_path, sep="\t", header=0, index_col=0)
    return df["Characteristics[ancestry category]"].to_dict()


def collate_gene_data(args, gene):
    """
    Write the following to an h5 file:
        genes: (n_samples,)
        seqs: (n_samples, 2, context_length, 4)
        samples: (n_samples,)
        ancestries: (n_samples,)
        Y: (n_samples,) counts
        Z: (n_samples,) z-scored counts
        P: (n_samples,) percentile of counts
    """
    output_path = os.path.join(args.output_dir, "tmp", f"{gene}.h5")
    if os.path.exists(output_path):
        return

    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    sample_to_ancestry_map = get_sample_ancestries(args.metadata_path)

    samples_with_counts = set(
        c for c in counts_df.columns if c.startswith("HG") or c.startswith("NA")
    )

    gene_seq_dir = os.path.join(args.consensus_seq_dir, gene)
    samples_with_seq = set(
        f.split(".")[0] for f in os.listdir(gene_seq_dir) if f != "ref.fa"
    )
    samples = list(sorted(samples_with_counts & samples_with_seq))
    assert len(samples) == 421, f"{gene} has {len(samples)} samples, expected 421"
    ancestries = [sample_to_ancestry_map[s] for s in samples]

    Y = counts_df.loc[gene, samples].to_numpy().astype(np.float32)
    assert not np.isnan(Y).any(), f"{gene} has NaNs in Y"
    Z = zscore(Y)
    P = percentileofscore(Y, Y)

    seqs = []
    for sample in samples:
        h1_fasta_path = os.path.join(gene_seq_dir, f"{sample}.1pIu.fa")
        h2_fasta_path = os.path.join(gene_seq_dir, f"{sample}.2pIu.fa")
        h1_seq = get_seq_from_fasta(h1_fasta_path, args.context_length)
        h2_seq = get_seq_from_fasta(h2_fasta_path, args.context_length)
        seqs.append(np.stack([h1_seq, h2_seq], axis=0))
    seqs = np.asarray(seqs)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("genes", data=[gene] * len(samples))
        f.create_dataset("seqs", data=seqs)
        f.create_dataset("samples", data=samples)
        f.create_dataset("ancestries", data=ancestries)
        f.create_dataset("Y", data=Y)
        f.create_dataset("Z", data=Z)
        f.create_dataset("P", data=P)


def split_dev_idxs(idxs: np.ndarray, frac_val: float = 0.05):
    np.random.shuffle(idxs)
    n_val = int(idxs.size * frac_val)
    val_idxs = idxs[:n_val]
    train_idxs = idxs[n_val:]
    return train_idxs, val_idxs


def flush_to_h5(input_f, output_f, idxs):
    idxs = np.sort(idxs)
    for dataset_name in input_f.keys():
        data = input_f[dataset_name][idxs]
        if dataset_name not in output_f:
            maxshape = (None,) + data.shape[1:]
            output_dataset = output_f.create_dataset(
                dataset_name, data=data, maxshape=maxshape
            )
        else:
            output_dataset = output_f[dataset_name]
            current_size = output_dataset.shape[0]
            output_dataset.resize(current_size + data.shape[0], axis=0)
            output_dataset[current_size:] = data
    output_f.flush()


def main():
    args = parse_args()
    tmp_dir = os.path.join(args.output_dir, "tmp")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    gene_to_class = get_genes(args)
    genes = list(gene_to_class.keys())
    publish_gene_class_csv(args.output_dir, gene_to_class)

    Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(collate_gene_data)(args, gene) for gene in genes
    )

    train_f = h5py.File(os.path.join(args.output_dir, "train.h5"), "w")
    val_f = h5py.File(os.path.join(args.output_dir, "val.h5"), "w")
    test_f = h5py.File(os.path.join(args.output_dir, "test.h5"), "w")

    for gene in tqdm(genes, desc="Flushing to joint h5"):
        gene_fpath = os.path.join(tmp_dir, f"{gene}.h5")

        with h5py.File(gene_fpath, "r") as in_f:
            ancestries = in_f["ancestries"][:].astype(str)

            if gene_to_class[gene] == "random_split":
                permutation = np.random.permutation(ancestries.size)
                test_idxs = permutation[:77]
                dev_idxs = permutation[77:]
                train_idxs, val_idxs = split_dev_idxs(dev_idxs)
            elif gene_to_class[gene] == "yri_split":
                test_idxs = np.where(ancestries == "Yoruba")[0]
                dev_idxs = np.where(ancestries != "Yoruba")[0]
                train_idxs, val_idxs = split_dev_idxs(dev_idxs)
            elif gene_to_class[gene] == "unseen":
                test_idxs = np.arange(ancestries.size)
                train_idxs, val_idxs = None, None
            else:
                raise ValueError(f"Unknown gene class {gene_to_class[gene]}")

            if train_idxs is not None:
                flush_to_h5(in_f, train_f, train_idxs)
            if val_idxs is not None:
                flush_to_h5(in_f, val_f, val_idxs)
            if test_idxs is not None:
                flush_to_h5(in_f, test_f, test_idxs)

    train_f.close()
    val_f.close()
    test_f.close()

    # Clean up
    for gene_fname in os.listdir(tmp_dir):
        gene_fpath = os.path.join(tmp_dir, gene_fname)
        os.remove(gene_fpath)
    os.rmdir(tmp_dir)


if __name__ == "__main__":
    main()
