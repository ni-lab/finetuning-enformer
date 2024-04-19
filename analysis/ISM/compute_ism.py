import os
import sys
from argparse import ArgumentParser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from enformer_pytorch import Enformer, str_to_one_hot
from pyfaidx import Fasta
from tqdm import tqdm

sys.path.append("/data/yosef3/users/ruchir/finetuning-enformer/predixcan_lite")
import utils
from genomic_utils.variant import Variant


class Dataset(torch.utils.data.Dataset):
    def __init__(self, ref_seq: str, bp_start: int, bp_end: int, variants: list):
        super().__init__()
        self.ref_seq = ref_seq
        self.bp_start = bp_start
        self.bp_end = bp_end
        self.variants = variants

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
        one_hot_alt_seq = str_to_one_hot(alt_seq)
        return {"seq": one_hot_alt_seq, "variant": str(v)}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--seqlen", type=int, default=384 * 128)
    parser.add_argument("--batch_size", type=int, default=32)
    # fmt: off
    parser.add_argument("--consensus_seq_dir", type=str, default="/data/yosef3/scratch/ruchir/data/basenji2_consensus_seqs")
    parser.add_argument("--counts_path", type=str, default="../../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--fasta_path", type=str, default="/data/yosef3/scratch/ruchir/data/genomes/hg19/hg19.fa")
    parser.add_argument("--gene_class_path", type=str, default="../../finetuning/data/h5_bins_384_chrom_split/gene_class.csv")
    # fmt: on
    return parser.parse_args()


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


@torch.no_grad()
def make_predictions(
    model, X, track_idx: int = 5110, avg_center_n_bins: int = 10
) -> np.ndarray:
    """
    X (tensor): (samples, length, 4)
    """
    outs = model(X)["human"][:, :, track_idx]

    # Average center bins
    assert outs.ndim == 2
    assert outs.shape[1] >= avg_center_n_bins
    bin_start = (outs.shape[1] - avg_center_n_bins) // 2
    bin_end = bin_start + avg_center_n_bins
    outs = outs[:, bin_start:bin_end].mean(dim=1)

    return outs.cpu().numpy()


def main():
    args = parse_args()

    genes = pd.read_csv(args.gene_class_path)["gene"].tolist()
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    samples = get_samples(counts_df, args.consensus_seq_dir, genes[0])

    fasta = Fasta(args.fasta_path)

    device = torch.device("cuda")
    model = Enformer.from_pretrained(
        "EleutherAI/enformer-official-rough", target_length=args.seqlen // 128
    ).to(device)

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
        one_hot_ref_seq = str_to_one_hot(ref_seq).unsqueeze(0).to(device)
        ref_pred = make_predictions(model, one_hot_ref_seq)[0]

        # Make predictions on variants
        ds = Dataset(ref_seq, bp_start, bp_end, genotype_mtx.index)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=4)
        variant_preds = {}
        for batch in tqdm(dl):
            seqs = batch["seq"].to(device)
            outs = make_predictions(model, seqs)
            for v_str, out in zip(batch["variant"], outs):
                v = Variant.create_from_str(v_str)
                variant_preds[v] = out
        variant_preds_l = [variant_preds.get(v, np.nan) for v in genotype_mtx.index]

        # Save predictions
        results_df = pd.DataFrame(
            {"ref": ref_pred, "variant": variant_preds_l}, index=genotype_mtx.index
        )
        results_df["ISM"] = results_df["variant"] - results_df["ref"]
        results_df.to_csv(os.path.join(args.output_dir, f"{gene}.csv"))


if __name__ == "__main__":
    main()
