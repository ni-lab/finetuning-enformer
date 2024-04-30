import operator
import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import reduce

import h5py
import numpy as np
import pandas as pd
from genomic_utils.variant import Variant
from joblib import Parallel, delayed
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_h5_path", type=str)
    parser.add_argument("--n_jobs", type=int, default=-1)
    # fmt: off
    parser.add_argument("--genotypes_dir", type=str, default="../genotypes/1Mb")
    parser.add_argument("--enformer_preds_dir", type=str, default="/data/yosef3/scratch/ruchir/data/enformer_hg19_predictions")
    # fmt: on
    return parser.parse_args()


def get_variants_by_chrom(genotypes_dir: str) -> dict[str, set[str]]:
    variants_by_chrom = defaultdict(set)
    for fname in tqdm(os.listdir(genotypes_dir), desc="Loading variants"):
        fpath = os.path.join(genotypes_dir, fname)
        df = pd.read_csv(fpath, header=0, usecols=[0], names=["variant"])
        variants = df["variant"].tolist()
        chrom = variants[0].split(":")[0].replace("chr", "")
        variants_by_chrom[chrom].update(variants)
    return variants_by_chrom


def load_enformer_scores_for_chrom(
    enformer_preds_dir: str, chrom: str, variants: set
) -> dict[str, np.ndarray]:
    h5_path = os.path.join(enformer_preds_dir, f"1000G.MAF_threshold=0.005.{chrom}.h5")

    scores = {}
    with h5py.File(h5_path, "r") as f:
        chr_ = f["chr"][:].astype(str)
        pos = f["pos"][:].astype(int)
        ref = f["ref"][:].astype(str)
        alt = f["alt"][:].astype(str)
        sar = f["SAR"][:].astype(np.float16)

        for i in range(len(chr_)):
            v_original = str(Variant(chr_[i], pos[i], ref[i], alt[i]))
            v_flipped = str(Variant(chr_[i], pos[i], alt[i], ref[i]))
            if v_original in variants:
                scores[v_original] = sar[i]
            if v_flipped in variants:
                scores[v_flipped] = sar[i]

    return scores


def main():
    args = parse_args()

    variants_by_chrom = get_variants_by_chrom(args.genotypes_dir)

    enformer_scores = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(load_enformer_scores_for_chrom)(
            args.enformer_preds_dir, chrom, variants
        )
        for chrom, variants in variants_by_chrom.items()
    )
    enformer_scores = reduce(operator.ior, enformer_scores, {})

    all_variants = sorted(set.union(*variants_by_chrom.values()))
    print(
        f"Loaded Enformer scores for {len(enformer_scores)}/{len(all_variants)} variants"
    )

    n_variants = len(all_variants)
    n_tracks = next(iter(enformer_scores.values())).size
    all_scores = np.full((n_variants, n_tracks), np.nan, dtype=np.float16)
    for i, v in enumerate(all_variants):
        if v in enformer_scores:
            all_scores[i] = enformer_scores[v]

    with h5py.File(args.output_h5_path, "w") as f:
        f.create_dataset("variants", data=all_variants)
        f.create_dataset("scores", data=all_scores)


if __name__ == "__main__":
    main()
