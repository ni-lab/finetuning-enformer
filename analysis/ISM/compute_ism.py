import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import torch
from Bio import Seq
from enformer_pytorch import Enformer, str_to_one_hot
from pyfaidx import Fasta
from tqdm import tqdm

from genomic_utils.variant import Variant

# Must import in this order because the finetuning directory also has a utils.py file
sys.path.append("/data/yosef3/users/ruchir/finetuning-enformer/vcf_utils") # "../../vcf_utils"
import utils
sys.path.append("/data/yosef3/users/ruchir/finetuning-enformer/finetuning") # "../../finetuning", 
import models


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, ref_seq: str, bp_start: int, bp_end: int, variants: list, rc: bool
    ):
        super().__init__()
        self.ref_seq = ref_seq
        self.bp_start = bp_start
        self.bp_end = bp_end
        self.variants = variants
        self.rc = rc

    def __len__(self):
        return len(self.variants)

    def __getitem__(self, idx):
        ref_seq = str(self.ref_seq)
        v = self.variants[idx]
        assert ref_seq[v.pos - self.bp_start] == v.ref

        alt_seq = (
            ref_seq[: v.pos - self.bp_start]
            + v.alt
            + ref_seq[v.pos - self.bp_start + 1 :]
        )

        if self.rc:
            alt_seq = str(Seq.Seq(alt_seq).reverse_complement())

        one_hot_alt_seq = str_to_one_hot(alt_seq)
        return {"seq": one_hot_alt_seq, "variant": str(v)}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "model_type", 
        type=str, 
        choices=["baseline", "classification", "regression", "single_regression"]
    )
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--seqlen", type=int, default=384 * 128)
    parser.add_argument("--batch_size", type=int, default=32)
    # fmt: off
    parser.add_argument("--consensus_seq_dir", type=str, default="/data/yosef3/scratch/ruchir/data/basenji2_consensus_seqs")
    parser.add_argument("--counts_path", type=str, default="/data/yosef3/users/ruchir/finetuning-enformer/process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--fasta_path", type=str, default="/data/yosef3/scratch/ruchir/data/genomes/hg19/hg19.fa")
    parser.add_argument("--gene_class_path", type=str, default="/data/yosef3/users/ruchir/finetuning-enformer/finetuning/data/h5_bins_384_chrom_split/gene_class.csv")
    # fmt: on
    args = parser.parse_args()

    if args.model_type != "baseline" and args.model_path is None:
        parser.error("--model_path is required for non-baseline models")

    return args


def get_samples(counts_df: pd.DataFrame, consensus_seq_dir: str, random_gene: str):
    samples_with_counts = set(
        [c for c in counts_df.columns if c.startswith("NA") or c.startswith("HG")]
    )

    gene_seq_dir = os.path.join(consensus_seq_dir, random_gene)
    samples_with_seq = set(
        [f.split(".")[0] for f in os.listdir(gene_seq_dir) if f != "ref.fa"]
    )

    samples = sorted(samples_with_counts & samples_with_seq)
    assert len(samples) == 421
    return samples


def load_model(model_type: str, model_path: str, seqlen: int):
    if model_type == "baseline":
        return Enformer.from_pretrained(
            "EleutherAI/enformer-official-rough", target_length=seqlen // 128
        )
    elif model_type == "classification":
        return models.PairwiseClassificationFloatPrecision.load_from_checkpoint(model_path)
    elif model_type == "regression":
        return models.PairwiseRegressionFloatPrecision.load_from_checkpoint(model_path)
    elif model_type == "single_regression":
        return models.SingleRegressionOnCountsFloatPrecision.load_from_checkpoint(model_path)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


@torch.no_grad()
def make_predictions(
    model,
    X,
    model_type: str,
    track_idx: int = 5110,
    avg_center_n_bins: int = 10,
) -> np.ndarray:
    """
    X (tensor): (samples, length, 4)
    """
    if model_type == "baseline":
        outs = model(X)["human"][:, :, track_idx]  # (samples, bins)
        assert outs.ndim == 2
        assert outs.shape[1] >= avg_center_n_bins

        bin_start = (outs.shape[1] - avg_center_n_bins) // 2
        bin_end = bin_start + avg_center_n_bins
        outs = outs[:, bin_start:bin_end].mean(dim=1)
        return outs.cpu().numpy()
    elif model_type in ["classification", "regression", "single_regression"]:
        outs = model(X, return_base_predictions=False, no_haplotype=True)
        assert outs.ndim == 1
        assert outs.shape[0] == X.shape[0]
        return outs.cpu().numpy()
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


def make_predictions_on_ref_seq(
    model, device, ref_seq: str, model_type: str, seqlen: int
) -> float:
    ref_seq_rc = str(Seq.Seq(ref_seq).reverse_complement())
    one_hot_ref_seq = str_to_one_hot(ref_seq)
    one_hot_ref_seq_rc = str_to_one_hot(ref_seq_rc)
    X = torch.stack([one_hot_ref_seq, one_hot_ref_seq_rc]).to(device)  # (2, length, 4)
    preds = make_predictions(model, X, model_type)  # (2,)
    return preds.mean()


def main():
    args = parse_args()

    genes = pd.read_csv(args.gene_class_path)["gene"].tolist()
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    samples = get_samples(counts_df, args.consensus_seq_dir, genes[0])
    fasta = Fasta(args.fasta_path)

    device = torch.device("cuda")
    model = load_model(args.model_type, args.model_path, args.seqlen).to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for gene in tqdm(genes):
        # Get genotype matrix
        genotype_mtx = utils.get_genotype_matrix(
            counts_df.loc[gene, "Chr"], counts_df.loc[gene, "Coord"], args.seqlen
        )

        # Remove variants that are not present in any of our samples
        genotype_mtx = genotype_mtx.loc[:, samples]
        allele_counts = utils.convert_to_dosage_matrix(genotype_mtx).sum(axis=1)
        genotype_mtx = genotype_mtx.loc[allele_counts > 0]

        # Load sequence from fasta
        chrom = counts_df.loc[gene, "Chr"]
        bp_start = counts_df.loc[gene, "Coord"] - args.seqlen // 2
        bp_end = bp_start + args.seqlen - 1
        ref_seq = fasta[f"chr{chrom}"][bp_start - 1 : bp_end].seq.upper()
        assert len(ref_seq) == args.seqlen

        # Make prediction on ref_seq
        ref_pred = make_predictions_on_ref_seq(
            model, device, ref_seq, args.model_type, args.seqlen
        )

        # Make predictions on variants
        forward_ds = Dataset(ref_seq, bp_start, bp_end, genotype_mtx.index, rc=False)
        reverse_ds = Dataset(ref_seq, bp_start, bp_end, genotype_mtx.index, rc=True)
        combined_ds = torch.utils.data.ConcatDataset([forward_ds, reverse_ds])
        combined_dl = torch.utils.data.DataLoader(
            combined_ds, batch_size=args.batch_size, num_workers=4
        )
        variant_preds = defaultdict(list)
        for batch in tqdm(combined_dl):
            seqs = batch["seq"].to(device)
            outs = make_predictions(model, seqs, args.model_type)
            for v_str, out in zip(batch["variant"], outs):
                v = Variant.create_from_str(v_str)
                variant_preds[v].append(out)

        assert all(len(v) == 2 for v in variant_preds.values())
        variant_preds = {v: np.mean(outs) for v, outs in variant_preds.items()}
        variant_preds_l = [variant_preds.get(v, np.nan) for v in genotype_mtx.index]

        # Save predictions
        results_df = pd.DataFrame(
            {"ref": ref_pred, "variant": variant_preds_l}, index=genotype_mtx.index
        )
        results_df["ISM"] = results_df["variant"] - results_df["ref"]
        results_df.to_csv(os.path.join(args.output_dir, f"{gene}.csv"))


if __name__ == "__main__":
    main()
