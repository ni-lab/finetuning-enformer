import os
import sys
from argparse import ArgumentParser

import h5py

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import torch
from enformer_pytorch import Enformer, str_to_one_hot
from pyfaidx import Fasta
from tqdm import tqdm

sys.path.append("../../finetuning/")
import models


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "model_type", type=str, choices=["baseline", "classification", "regression"]
    )
    parser.add_argument("output_path", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--seqlen", type=int, default=384 * 128)
    # fmt: off
    parser.add_argument("--counts_path", type=str, default="../../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--fasta_path", type=str, default="/data/yosef3/scratch/ruchir/data/genomes/hg19/hg19.fa")
    parser.add_argument("--gene_class_path", type=str, default="../../finetuning/data/h5_bins_384_chrom_split/gene_class.csv")
    # fmt: on
    args = parser.parse_args()

    if args.model_type != "baseline" and args.model_path is None:
        parser.error("--model_path is required for non-baseline models")

    return args


def load_model(model_type: str, model_path: str, seqlen: int):
    if model_type == "baseline":
        return Enformer.from_pretrained(
            "EleutherAI/enformer-official-rough", target_length=seqlen // 128
        )
    elif model_type == "classification":
        raise NotImplementedError
    elif model_type == "regression":
        return models.PairwiseRegressionFloatPrecision.load_from_checkpoint(model_path)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


def make_predictions(
    model, X, model_type: str, track_idx: int = 5110, avg_center_n_bins: int = 10
):
    """
    X (tensor): (N, L, 4)
    """
    if model_type == "baseline":
        outs = model(X)["human"][:, :, track_idx]  # (samples, bins)
        assert outs.ndim == 2
        assert outs.shape[1] >= avg_center_n_bins

        bin_start = (outs.shape[1] - avg_center_n_bins) // 2
        bin_end = bin_start + avg_center_n_bins
        outs = outs[:, bin_start:bin_end].mean(dim=1)
        return outs

    elif model_type == "classification":
        raise NotImplementedError

    elif model_type == "regression":
        outs = model.base(
            X, return_only_embeddings=True, target_length=model.hparams.n_total_bins
        )  # (samples, bins, embedding_dim)

        bin_start = (outs.shape[1] - avg_center_n_bins) // 2
        bin_end = bin_start + avg_center_n_bins
        outs = outs[:, bin_start:bin_end, :]  # (samples, center_bins, embedding_dim)

        outs = model.attention_pool(outs)  # (samples, embedding_dim)
        outs = model.prediction_head(outs)  # (samples, 1)
        outs = outs.squeeze(1)  # (samples,)
        return outs


def main():
    args = parse_args()
    genes = pd.read_csv(args.gene_class_path)["gene"].tolist()
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    fasta = Fasta(args.fasta_path)

    device = torch.device("cuda")
    model = load_model(args.model_type, args.model_path, args.seqlen).to(device)
    model.eval()

    all_one_hot_seqs = np.full((len(genes), args.seqlen, 4), np.nan)
    all_gradients = np.full((len(genes), args.seqlen, 4), np.nan)
    bp_starts = [None for _ in genes]
    bp_ends = [None for _ in genes]

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
        preds = make_predictions(model, one_hot_seq, args.model_type)
        gradient = torch.autograd.grad(preds, one_hot_seq)[0]  # (1, L, 4)
        assert gradient.shape == one_hot_seq.shape

        all_one_hot_seqs[i] = one_hot_seq.detach().cpu().numpy()[0]
        all_gradients[i] = gradient.detach().cpu().numpy()[0]
        bp_starts[i] = bp_start
        bp_ends[i] = bp_end

    assert not np.isnan(all_one_hot_seqs).any()
    assert not np.isnan(all_gradients).any()
    assert all(bp_start is not None for bp_start in bp_starts)
    assert all(bp_end is not None for bp_end in bp_ends)

    # Write to h5 file
    with h5py.File(args.output_path, "w") as f:
        f.create_dataset("genes", data=genes)
        f.create_dataset("seqs", data=all_one_hot_seqs)
        f.create_dataset("gradients", data=all_gradients)
        f.create_dataset("bp_starts", data=bp_starts)
        f.create_dataset("bp_ends", data=bp_ends)


if __name__ == "__main__":
    main()
