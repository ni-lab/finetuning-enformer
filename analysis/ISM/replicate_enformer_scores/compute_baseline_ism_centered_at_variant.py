import os
import sys
from argparse import ArgumentParser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import pandas as pd
import torch
from Bio import Seq
from enformer_pytorch import Enformer, str_to_one_hot
from pyfaidx import Fasta
from tqdm import tqdm

sys.path.append("/data/yosef3/users/ruchir/finetuning-enformer/predixcan_lite")
import utils
from genomic_utils.variant import Variant


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        variants: list[Variant],
        fasta_path: str,
        seqlen: int,
        use_ref: bool,
        rc: bool = False,
    ):
        super().__init__()
        self.variants = variants
        self.fasta = Fasta(fasta_path)
        self.seqlen = seqlen
        self.use_ref = use_ref
        self.rc = rc

    def __len__(self):
        return len(self.variants)

    def __getitem__(self, idx):
        v = self.variants[idx]
        bp_start = v.pos - self.seqlen // 2
        bp_end = bp_start + self.seqlen - 1
        seq = self.fasta[f"chr{v.chrom}"][bp_start - 1 : bp_end].seq.upper()
        assert (
            len(seq) == self.seqlen
        ), f"{len(seq)} != {self.seqlen} for variant {v}, bp_start: {bp_start}, bp_end: {bp_end}"

        if self.use_ref:
            assert (
                seq[v.pos - bp_start] == v.ref
            ), f"{seq[v.pos - bp_start]} != {v.ref} for variant {v} and use_ref=True"
        else:
            seq = seq[: v.pos - bp_start] + v.alt + seq[v.pos - bp_start + 1 :]
            assert (
                seq[v.pos - bp_start] == v.alt
            ), f"{seq[v.pos - bp_start]} != {v.alt} for variant {v} and use_ref=False"

        if self.rc:
            seq = str(Seq.Seq(seq).reverse_complement())

        return {"seq": str_to_one_hot(seq), "variant": str(v)}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--context_size", type=int, default=384 * 128)
    parser.add_argument("--seqlen", type=int, default=196_608)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_genes", type=int, default=1)
    # fmt: off
    parser.add_argument("--counts_path", type=str, default="/data/yosef3/users/ruchir/finetuning-enformer/process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--fasta_path", type=str, default="/data/yosef3/scratch/ruchir/data/genomes/hg19/hg19.fa")
    parser.add_argument("--gene_class_path", type=str, default="/data/yosef3/users/ruchir/finetuning-enformer/finetuning/data/h5_bins_384_chrom_split/gene_class.csv")
    # fmt: on
    return parser.parse_args()


@torch.no_grad()
def make_predictions(model, X, track_idx: int = 5110) -> np.ndarray:
    """
    X (tensor): (samples, length, 4)
    """
    outs = model(X)["human"][:, :, track_idx]  # (samples, bins)
    assert outs.shape[1] == 896
    outs = outs.mean(dim=1)  # (samples,)
    return outs.cpu().numpy()


def make_predictions_on_variants(
    model, variants, fasta_path, seqlen, use_ref, rc, batch_size, device
):
    ds = Dataset(variants, fasta_path, seqlen, use_ref, rc)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=1)
    variant_preds = {}
    for batch in tqdm(dl):
        seqs = batch["seq"].to(device)
        outs = make_predictions(model, seqs)
        for v_str, out in zip(batch["variant"], outs):
            v = Variant.create_from_str(v_str)
            variant_preds[v] = out
    return variant_preds


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    genes = pd.read_csv(args.gene_class_path)["gene"].tolist()
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")

    device = torch.device("cuda")
    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough").to(device)

    for i in range(args.n_genes):
        gene = genes[i]

        genotype_mtx = utils.get_genotype_matrix(
            counts_df.loc[gene, "Chr"], counts_df.loc[gene, "Coord"], args.context_size
        )
        variants = genotype_mtx.index.tolist()

        ref_forward_preds = make_predictions_on_variants(
            model,
            variants,
            args.fasta_path,
            args.seqlen,
            True,
            False,
            args.batch_size,
            device,
        )
        ref_reverse_preds = make_predictions_on_variants(
            model,
            variants,
            args.fasta_path,
            args.seqlen,
            True,
            True,
            args.batch_size,
            device,
        )

        alt_forward_preds = make_predictions_on_variants(
            model,
            variants,
            args.fasta_path,
            args.seqlen,
            False,
            False,
            args.batch_size,
            device,
        )
        alt_reverse_preds = make_predictions_on_variants(
            model,
            variants,
            args.fasta_path,
            args.seqlen,
            False,
            True,
            args.batch_size,
            device,
        )

        results_df = pd.DataFrame(
            {
                "ref_forward": ref_forward_preds,
                "ref_reverse": ref_reverse_preds,
                "alt_forward": alt_forward_preds,
                "alt_reverse": alt_reverse_preds,
            }
        )
        results_df["ref"] = 0.5 * (
            results_df["ref_forward"] + results_df["ref_reverse"]
        )
        results_df["alt"] = 0.5 * (
            results_df["alt_forward"] + results_df["alt_reverse"]
        )
        results_df["ISM_forward"] = (
            results_df["alt_forward"] - results_df["ref_forward"]
        )
        results_df["ISM_reverse"] = (
            results_df["alt_reverse"] - results_df["ref_reverse"]
        )
        results_df["ISM"] = results_df["alt"] - results_df["ref"]

        results_df.to_csv(os.path.join(args.output_dir, f"{gene}.csv"))


if __name__ == "__main__":
    main()
