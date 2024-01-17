import operator
import os
from functools import reduce

import h5py
import numpy as np
import pandas as pd
from genomic_utils.variant import Variant
from joblib import Parallel, delayed
from tqdm import tqdm


def load_gene_expression_counts(counts_path: str) -> pd.DataFrame:
    counts_df = pd.read_csv(counts_path, index_col=0)
    counts_df = counts_df.reset_index().set_index("our_gene_name")
    counts_df = counts_df.loc[counts_df.index.dropna()]
    return counts_df


def load_sample_ancestries(
    geuvadis_metadata_path: str = "/data/yosef3/users/ruchir/pgp_uq/data/E-GEUV-1.sdrf.txt",
) -> dict[str, str]:
    metadata_df = pd.read_csv(geuvadis_metadata_path, sep="\t", header=0, index_col=0)
    return metadata_df["Characteristics[ancestry category]"].to_dict()


def _load_sar_scores_for_chrom(
    chrom: str,
    variants: set[str],
    track_idxs: list[int],
    track_weights: list[int],
    enformer_predictions_dir: str = "/data/yosef3/scratch/ruchir/data/enformer_hg19_predictions",
) -> dict[str, float]:
    sar_scores = {}
    h5_path = os.path.join(
        enformer_predictions_dir, f"1000G.MAF_threshold=0.005.{chrom}.h5"
    )

    with h5py.File(h5_path, "r") as f:
        chr = f["chr"][:].astype(str)
        pos = f["pos"][:].astype(int)
        ref = f["ref"][:].astype(str)
        alt = f["alt"][:].astype(str)
        scores = np.abs(f["SAR"][:, track_idxs].astype(float))
        weighted_scores = scores @ np.asarray(track_weights)

        for i in range(len(chr)):
            # Check both major/minor and minor/major because we don't know which
            # corresponds to hg19_ref/hg19_alt
            original_v = str(Variant(chr[i], pos[i], ref[i], alt[i]))
            flipped_v = str(Variant(chr[i], pos[i], alt[i], ref[i]))
            if original_v in variants:
                sar_scores[original_v] = weighted_scores[i]
            if flipped_v in variants:
                sar_scores[flipped_v] = weighted_scores[i]

    return sar_scores


def load_sar_scores(
    dosage_matrices_dir: str,
    track_idxs: int = [69, 5110],
    track_weights: list[int] = [0.5, 0.5],
    n_jobs: int = -1,
) -> dict[str, float]:
    """
    Load |SAR| scores for all variants in dosage matrices.
    """

    # 1. Get variants for which we need scores from dosage matrices
    variants = set()
    for fname in tqdm(os.listdir(dosage_matrices_dir), desc="Loading variants"):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(dosage_matrices_dir, fname)
        gene_df = pd.read_csv(fpath, header=0, usecols=[0], names=["variant"])
        variants.update(gene_df["variant"].tolist())
    print(f"Loaded {len(variants)} variants from dosage matrices")

    # 2. Load |SAR| scores
    sar_scores = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_load_sar_scores_for_chrom)(chrom, variants, track_idxs, track_weights)
        for chrom in range(1, 23)
    )
    sar_scores = reduce(operator.ior, sar_scores, {})
    print(f"Loaded SAR scores for {len(sar_scores)}/{len(variants)} variants")
    return sar_scores
