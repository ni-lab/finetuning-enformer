import os
from argparse import ArgumentParser

import h5py

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import pandas as pd
import torch
from enformer_pytorch import Enformer, str_to_one_hot
from pyfaidx import Fasta
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_path", type=str)
    parser.add_argument("--seqlen", type=int, default=384 * 128)
    # fmt: off
    parser.add_argument("--counts_path", type=str, default="../../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--fasta_path", type=str, default="/data/yosef3/scratch/ruchir/data/genomes/hg19/hg19.fa")
    parser.add_argument("--gene_class_path", type=str, default="../../finetuning/data/h5_bins_384_chrom_split/gene_class.csv")
    # fmt: on
    return parser.parse_args()


def make_predictions(model, X, track_idx: int = 5110, avg_center_n_bins: int = 10):
    """
    X (tensor): (N, L, 4)
    """
    outs = model(X)["human"][:, :, track_idx]  # (N, B)
    assert outs.ndim == 2
    assert outs.shape[1] >= avg_center_n_bins

    # Average the center bins
    bin_start = (outs.shape[1] - avg_center_n_bins) // 2
    bin_end = bin_start + avg_center_n_bins
    outs = outs[:, bin_start:bin_end].mean(dim=1)  # (N,)
    return outs


def main():
    args = parse_args()
    genes = pd.read_csv(args.gene_class_path)["gene"].tolist()
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    fasta = Fasta(args.fasta_path)

    device = torch.device("cuda")
    model = Enformer.from_pretrained(
        "EleutherAI/enformer-official-rough", target_length=args.seqlen // 128
    ).to(device)
    model.eval()

    all_one_hot_seqs = np.full((len(genes), args.seqlen, 4), np.nan)
    all_gradients = np.full((len(genes), args.seqlen, 4), np.nan)

    for i, gene in tqdm(enumerate(genes)):
        chrom = counts_df.loc[gene, "Chr"]
        bp_start = counts_df.loc[gene, "Coord"] - args.seqlen // 2
        bp_end = bp_start + args.seqlen - 1
        seq = fasta[f"chr{chrom}"][bp_start - 1 : bp_end].seq.upper()
        assert len(seq) == args.seqlen

        one_hot_seq = str_to_one_hot(seq)  # (L, 4)
        one_hot_seq = one_hot_seq.unsqueeze(0).to(device)  # (1, L, 4)
        one_hot_seq.requires_grad_(True)

        # Compute gradient of the prediction with respect to the input
        model.zero_grad()
        preds = make_predictions(model, one_hot_seq)
        gradient = torch.autograd.grad(preds, one_hot_seq)[0]  # (1, L, 4)
        assert gradient.shape == one_hot_seq.shape

        all_one_hot_seqs[i] = one_hot_seq.detach().cpu().numpy()[0]
        all_gradients[i] = gradient.detach().cpu().numpy()[0]

    assert not np.isnan(all_one_hot_seqs).any()
    assert not np.isnan(all_gradients).any()

    # Write to h5 file
    with h5py.File(args.output_path, "w") as f:
        f.create_dataset("genes", data=genes)
        f.create_dataset("seqs", data=all_one_hot_seqs)
        f.create_dataset("gradients", data=all_gradients)


if __name__ == "__main__":
    main()
