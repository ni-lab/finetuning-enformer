"""Create an h5 file with the remainder of the unseen genes.

In construct_h5_dataset.py, we created a dataset consisting of 200 random-split genes, 200
population-split genes, and 100 unseen genes. There are more genes with a European eQTL though. We
collect the remainder of those unseen genes in this script.
"""

import os
import shutil
from argparse import ArgumentParser

import construct_h5_dataset
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


# fmt: off
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--test_h5_path", type=str, required=True, help="Path to the test h5 file from previously constructed dataset")
    parser.add_argument("--output_h5_path", type=str, required=True, help="Path to the output h5 file")
    parser.add_argument("--context_length", type=int, default=128 * 384)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--counts_path", type=str, default="/data/yosef3/users/ruchir/finetuning-enformer/process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--consensus_seq_dir", type=str, default="/data/yosef3/scratch/ruchir/data/basenji2_consensus_seqs")
    parser.add_argument("--metadata_path", type=str, default="/data/yosef3/users/ruchir/pgp_uq/data/E-GEUV-1.sdrf.txt")
    parser.add_argument("--tmp_dir", type=str, default="tmp")
    return parser.parse_args()
# fmt: on


def get_remaining_unseen_genes(test_h5_path: str, counts_path: str) -> list[str]:
    # First, get list of genes with a European eQTL
    counts_df = pd.read_csv(counts_path, index_col="our_gene_name")
    counts_df = counts_df[counts_df["EUR_eGene"]].copy()
    assert counts_df.index.notnull().all()
    all_genes = set(counts_df.index)

    # Next, get list of genes in the previously constructed test h5 file
    previous_genes = None
    with h5py.File(test_h5_path, "r") as f:
        genes = f["genes"][:].astype(str)
        previous_genes = set(genes)

    remaining_genes = all_genes - previous_genes
    return sorted(remaining_genes)


def main():
    args = parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)

    remaining_genes = get_remaining_unseen_genes(args.test_h5_path, args.counts_path)
    print(f"Found {len(remaining_genes)} remaining unseen genes")

    # Write gene-specific data to temporary h5 files
    Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(construct_h5_dataset.collate_gene_data)(
            gene,
            args.tmp_dir,
            args.counts_path,
            args.metadata_path,
            args.consensus_seq_dir,
            args.context_length,
        )
        for gene in remaining_genes
    )

    # Construct one h5 file from all the temporary h5 files
    out_f = h5py.File(args.output_h5_path, "w")
    for gene in tqdm(remaining_genes):
        gene_h5_path = os.path.join(args.tmp_dir, f"{gene}.h5")
        with h5py.File(gene_h5_path, "r") as in_f:
            test_idxs = np.arange(in_f["samples"].shape[0])
            construct_h5_dataset.flush_to_h5(in_f, out_f, test_idxs)

    # Clean up
    shutil.rmtree(args.tmp_dir)


if __name__ == "__main__":
    main()
