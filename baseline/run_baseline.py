import os
from argparse import ArgumentParser
from collections import defaultdict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import h5py
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from enformer_pytorch import Enformer
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, seqlen: int, rc: bool = False):
        super().__init__()
        assert seqlen % 128 == 0

        self.h5_file = h5py.File(h5_path, "r")
        self.seqlen = seqlen
        self.rc = rc

        self.genes = self.h5_file["genes"][:].astype(str)
        self.samples = self.h5_file["samples"][:].astype(str)
        self.seqs = self.h5_file["seqs"]  # (n_seqs, 2, length, 4)
        assert self.genes.size == self.samples.size == self.seqs.shape[0]

    def __len__(self):
        return self.genes.size

    def __shorten_seq(self, seq):
        """
        seq: (2, seqlen, 4)
        """
        if seq.shape[1] == self.seqlen:
            return seq
        start_idx = (seq.shape[1] - self.seqlen) // 2
        end_idx = start_idx + self.seqlen
        return seq[:, start_idx:end_idx, :]

    def __getitem__(self, idx):
        seq = self.__shorten_seq(self.seqs[idx]).astype(np.float32)
        if self.rc:
            seq = np.flip(seq, axis=(-1, -2)).copy()
        gene = self.genes[idx]
        sample = self.samples[idx]
        return {"seq": seq, "gene": gene, "sample": sample}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_path", type=str)
    # fmt: off
    parser.add_argument("--h5_dir", type=str, default="../finetuning/data/h5_bins_384_chrom_split")
    # fmt: on
    parser.add_argument("--seqlen", type=int, default=384 * 128)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    assert args.seqlen % 128 == 0
    return args


def average_center_bins(X, avg_center_n_bins: int):
    """
    X (tensor): (seqs, bins)
    """
    assert X.ndim == 2
    assert X.shape[1] >= avg_center_n_bins

    bin_start = (X.shape[1] - avg_center_n_bins) // 2
    bin_end = bin_start + avg_center_n_bins
    return X[:, bin_start:bin_end].mean(dim=1)


@torch.no_grad()
def make_predictions(
    model, X, track_idx: int = 5110, avg_center_n_bins: int = 10
) -> np.ndarray:
    """
    X (tensor): (samples, haplotypes, length, 4)
    """
    X = rearrange(X, "S H L BP -> (S H) L BP")
    outs = model(X)["human"][:, :, track_idx]  # (samples * haplotypes, target_length)
    outs = average_center_bins(outs, avg_center_n_bins)
    outs = rearrange(outs, "(S H) -> S H", H=2)
    outs = outs.mean(dim=1).cpu().numpy()
    return outs


def get_all_preds(model, device, ds, preds, batch_size):
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=8
    )
    for batch in tqdm(dl):
        seqs = batch["seq"].to(device)
        genes = batch["gene"]
        samples = batch["sample"]
        outs = make_predictions(model, seqs)
        for gene, sample, out in zip(genes, samples, outs):
            preds[gene][sample].append(out)


def main():
    args = parse_args()

    device = torch.device("cuda")
    model = Enformer.from_pretrained(
        "EleutherAI/enformer-official-rough", target_length=args.seqlen // 128
    ).to(device)
    model.eval()

    preds = defaultdict(
        lambda: defaultdict(list)
    )  # gene -> sample -> list of preds (one for forward, one for reverse)
    for split in ["train", "val", "test"]:
        for rc in [False, True]:
            ds = Dataset(
                os.path.join(args.h5_dir, f"{split}.h5"), seqlen=args.seqlen, rc=rc
            )
            get_all_preds(model, device, ds, preds, args.batch_size)

    for g in preds:
        for s in preds[g]:
            assert len(preds[g][s]) == 2
            preds[g][s] = np.mean(preds[g][s])

    # Save predictions as a dataframe
    genes = sorted(preds.keys())
    samples = sorted({s for gene in preds for s in preds[gene]})
    preds_mtx = np.full((len(genes), len(samples)), np.nan)
    for row, g in enumerate(genes):
        for col, s in enumerate(samples):
            preds_mtx[row, col] = preds[g][s]

    preds_df = pd.DataFrame(preds_mtx, index=genes, columns=samples)
    preds_df.to_csv(args.output_path)


if __name__ == "__main__":
    main()
