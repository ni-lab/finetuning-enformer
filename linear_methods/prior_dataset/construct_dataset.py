import os
import sys
from argparse import ArgumentParser
from functools import partial

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append("/data/yosef3/users/ruchir/finetuning-enformer/predixcan_lite")
import utils
from genomic_utils.variant import Variant


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("data_split_path")
    parser.add_argument("--context_size", type=int, default=384 * 128)
    parser.add_argument("--n_jobs", type=int, default=-1)
    # fmt: off
    parser.add_argument("--counts_path", type=str, default="../../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--meta_features_h5_path", type=str, default="../data/meta_features/enformer_scores.1Mb.h5")
    # fmt: on
    return parser.parse_args()


def load_splits_per_gene(data_split_path: str) -> tuple[dict, dict, dict]:
    splits_df = pd.read_csv(data_split_path, index_col=0)  # (genes, samples)

    train_samples = {}
    val_samples = {}
    test_samples = {}
    for gene in splits_df.index:
        row = splits_df.loc[gene]
        train_samples[gene] = row[row == "train"].index.tolist()
        val_samples[gene] = row[row == "val"].index.tolist()
        test_samples[gene] = row[row == "test"].index.tolist()
    return (train_samples, val_samples, test_samples)


def load_dosages_per_gene(
    counts_df: pd.DataFrame,
    train_samples_per_gene: dict,
    val_samples_per_gene: dict,
    test_samples_per_gene: dict,
    context_size: int,
    n_jobs: int,
):
    def load_gene_dosages(
        gene: str,
        train_samples: list[str],
        val_samples: list[str],
        test_samples: list[str],
    ):
        genotype_mtx = utils.get_genotype_matrix(
            counts_df.loc[gene, "Chr"], counts_df.loc[gene, "Coord"], context_size
        )  # [variants, samples]

        train_genotype_mtx = genotype_mtx[train_samples]  # (variants, samples)
        train_genotype_mtx = utils.filter_common(train_genotype_mtx)
        train_genotype_mtx = utils.filter_hwe(train_genotype_mtx)

        val_genotype_mtx = genotype_mtx.loc[train_genotype_mtx.index, val_samples]
        test_genotype_mtx = genotype_mtx.loc[train_genotype_mtx.index, test_samples]

        train_dosage_mtx = (
            utils.convert_to_dosage_matrix(train_genotype_mtx).to_numpy().T
        )
        val_dosage_mtx = utils.convert_to_dosage_matrix(val_genotype_mtx).to_numpy().T
        test_dosage_mtx = utils.convert_to_dosage_matrix(test_genotype_mtx).to_numpy().T

        if train_dosage_mtx.shape[1] == 0:
            print(f"Skipping dosage normalization for gene {gene} due to no variants.")
        else:
            scaler = StandardScaler()
            train_dosage_mtx = scaler.fit_transform(train_dosage_mtx)
            val_dosage_mtx = scaler.transform(val_dosage_mtx)
            test_dosage_mtx = scaler.transform(test_dosage_mtx)

        return (
            train_dosage_mtx,
            val_dosage_mtx,
            test_dosage_mtx,
            train_genotype_mtx.index.tolist(),
        )

    genes = list(train_samples_per_gene.keys())
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(load_gene_dosages)(
            gene,
            train_samples_per_gene[gene],
            val_samples_per_gene[gene],
            test_samples_per_gene[gene],
        )
        for gene in genes
    )
    train_dosages = {g: r[0] for g, r in zip(genes, results)}
    val_dosages = {g: r[1] for g, r in zip(genes, results)}
    test_dosages = {g: r[2] for g, r in zip(genes, results)}
    variants = {g: r[3] for g, r in zip(genes, results)}
    return (train_dosages, val_dosages, test_dosages, variants)


def load_meta_features_per_gene(
    variants_per_gene: dict[str, list[Variant]],
    meta_features_h5_path: str,
):
    meta_features_per_gene = {}

    with h5py.File(meta_features_h5_path, "r") as f:
        variants = f["variants"][:].astype(str)
        variant_to_idx = {v: i for i, v in enumerate(variants)}
        scores = np.abs(
            f["scores"][:].astype(np.float16)
        )  # (n_variants, n_meta_features)
        mean_scores = np.nanmean(scores, axis=0)  # (n_meta_features,)

        for g in tqdm(variants_per_gene, desc="Loading meta features"):
            idxs = [variant_to_idx[str(v)] for v in variants_per_gene[g]]
            meta_features = scores[idxs]

            # Replace NaNs with mean scores
            nan_mask = np.isnan(meta_features)
            meta_features[nan_mask] = mean_scores[np.nonzero(nan_mask)[1]]

            meta_features_per_gene[g] = meta_features

    return meta_features_per_gene


def load_Y(
    counts_df: pd.DataFrame,
    train_samples_per_gene: dict,
    val_samples_per_gene: dict,
    test_samples_per_gene: dict,
):
    train_Y = {}
    val_Y = {}
    test_Y = {}

    for g in train_samples_per_gene:
        train_Y[g] = counts_df.loc[g, train_samples_per_gene[g]].to_numpy()
        val_Y[g] = counts_df.loc[g, val_samples_per_gene[g]].to_numpy()
        test_Y[g] = counts_df.loc[g, test_samples_per_gene[g]].to_numpy()

    return (train_Y, val_Y, test_Y)


def load_z_scores(
    counts_df: pd.DataFrame,
    train_samples_per_gene: dict,
    val_samples_per_gene: dict,
    test_samples_per_gene: dict,
):
    train_z_scores = {}
    val_z_scores = {}
    test_z_scores = {}

    for g in train_samples_per_gene:
        train_counts = counts_df.loc[g, train_samples_per_gene[g]].to_numpy()
        val_counts = counts_df.loc[g, val_samples_per_gene[g]].to_numpy()
        test_counts = counts_df.loc[g, test_samples_per_gene[g]].to_numpy()

        scaler = StandardScaler()
        train_z_scores[g] = scaler.fit_transform(train_counts.reshape(-1, 1)).flatten()
        val_z_scores[g] = scaler.transform(val_counts.reshape(-1, 1)).flatten()
        test_z_scores[g] = scaler.transform(test_counts.reshape(-1, 1)).flatten()

    return (train_z_scores, val_z_scores, test_z_scores)


def publish_dataset(
    output_h5_path: str,
    samples_per_gene: dict,
    variants_per_gene: dict,
    dosages_per_gene: dict,
    meta_features_per_gene: dict,
    Y_per_gene: dict,
    z_scores_per_gene: dict,
):
    print(f"Publishing dataset to {output_h5_path}")
    with h5py.File(output_h5_path, "w") as f:
        genes = list(samples_per_gene.keys())
        f.create_dataset("genes", data=genes)
        for g in tqdm(genes):
            f.create_group(g)
            f[g].create_dataset("samples", data=samples_per_gene[g])
            f[g].create_dataset("variants", data=[str(v) for v in variants_per_gene[g]])
            f[g].create_dataset("dosages", data=dosages_per_gene[g].astype(np.float16))
            f[g].create_dataset(
                "meta_features", data=meta_features_per_gene[g].astype(np.float16)
            )
            f[g].create_dataset("Y", data=Y_per_gene[g].astype(np.float16))
            f[g].create_dataset(
                "z_scores", data=z_scores_per_gene[g].astype(np.float16)
            )


def main():
    args = parse_args()

    # Load data
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    train_samples, val_samples, test_samples = load_splits_per_gene(
        args.data_split_path
    )
    train_dosages, val_dosages, test_dosages, variants = load_dosages_per_gene(
        counts_df,
        train_samples,
        val_samples,
        test_samples,
        args.context_size,
        args.n_jobs,
    )
    meta_features = load_meta_features_per_gene(variants, args.meta_features_h5_path)
    train_Y, val_Y, test_Y = load_Y(
        counts_df,
        train_samples,
        val_samples,
        test_samples,
    )
    train_z_scores, val_z_zcores, test_z_scores = load_z_scores(
        counts_df,
        train_samples,
        val_samples,
        test_samples,
    )

    # Publish dataset
    os.makedirs(args.output_dir, exist_ok=True)
    publisher = partial(
        publish_dataset,
        variants_per_gene=variants,
        meta_features_per_gene=meta_features,
    )
    publisher(
        output_h5_path=os.path.join(args.output_dir, "train.h5"),
        samples_per_gene=train_samples,
        dosages_per_gene=train_dosages,
        Y_per_gene=train_Y,
        z_scores_per_gene=train_z_scores,
    )
    publisher(
        output_h5_path=os.path.join(args.output_dir, "val.h5"),
        samples_per_gene=val_samples,
        dosages_per_gene=val_dosages,
        Y_per_gene=val_Y,
        z_scores_per_gene=val_z_zcores,
    )
    publisher(
        output_h5_path=os.path.join(args.output_dir, "test.h5"),
        samples_per_gene=test_samples,
        dosages_per_gene=test_dosages,
        Y_per_gene=test_Y,
        z_scores_per_gene=test_z_scores,
    )


if __name__ == "__main__":
    main()
