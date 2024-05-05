import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
import utils
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str)
    # fmt: off
    parser.add_argument("--counts_path", type=str, default="../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--sample_splits_path", type=str, default="h5_bins_384_chrom_split/sample_splits.h5_bins_384_chrom_split.csv")
    # fmt: on
    parser.add_argument("--context_size", type=int, default=1_000_000)
    parser.add_argument("--chunk_start", type=int, default=0)
    parser.add_argument("--chunk_end", type=int, default=None)
    return parser.parse_args()


def load_counts_and_sample_splits(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load sample splits and subset to genes within the specified chunk
    sample_splits_df = pd.read_csv(args.sample_splits_path, index_col=0)
    chunk_end = len(sample_splits_df) if args.chunk_end is None else args.chunk_end
    sample_splits_df = sample_splits_df.iloc[args.chunk_start : chunk_end]

    # Load counts and subset to genes within the specified chunk
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    counts_df = counts_df.loc[sample_splits_df.index]
    return (counts_df, sample_splits_df)


def load_gene_data(
    gene: str,
    counts_df: pd.DataFrame,
    sample_splits_df: pd.DataFrame,
    context_size: int,
):
    """
    Returns
        train_data (dict[str, np.ndarray])
        val_data (dict[str, np.ndarray])
        test_data (dict[str, np.ndarray])
        variants (list[Variant])
    """
    train_data, val_data, test_data = {}, {}, {}

    # Get train, val, and test samples
    sample_splits = sample_splits_df.loc[gene]
    train_data["samples"] = sample_splits[sample_splits == "train"].index.tolist()
    val_data["samples"] = sample_splits[sample_splits == "val"].index.tolist()
    test_data["samples"] = sample_splits[sample_splits == "test"].index.tolist()

    # Load genotype matrix and filter variants by MAF/ HWE
    genotype_mtx = utils.get_genotype_matrix(
        counts_df.loc[gene, "Chr"], counts_df.loc[gene, "Coord"], context_size
    )  # [variants, samples]

    train_genotype_mtx = genotype_mtx[train_data["samples"]]
    train_genotype_mtx = utils.filter_common(train_genotype_mtx)
    train_genotype_mtx = utils.filter_hwe(train_genotype_mtx)
    val_genotype_mtx = genotype_mtx.loc[train_genotype_mtx.index, val_data["samples"]]
    test_genotype_mtx = genotype_mtx.loc[train_genotype_mtx.index, test_data["samples"]]

    # Create X and Y matrices
    train_dosage_mtx = utils.convert_to_dosage_matrix(train_genotype_mtx)
    val_dosage_mtx = utils.convert_to_dosage_matrix(val_genotype_mtx)
    test_dosage_mtx = utils.convert_to_dosage_matrix(test_genotype_mtx)

    train_data["X"] = train_dosage_mtx.to_numpy(dtype=np.float64).T
    val_data["X"] = val_dosage_mtx.to_numpy(dtype=np.float64).T
    test_data["X"] = test_dosage_mtx.to_numpy(dtype=np.float64).T

    train_data["Y"] = counts_df.loc[gene, train_data["samples"]].to_numpy()
    val_data["Y"] = counts_df.loc[gene, val_data["samples"]].to_numpy()
    test_data["Y"] = counts_df.loc[gene, test_data["samples"]].to_numpy()

    return (train_data, val_data, test_data, train_dosage_mtx.index.tolist())


@ignore_warnings(category=ConvergenceWarning)
def train_elastic_net_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    l1_ratio: float = 0.5,
    max_iter: int = 1_000,
):
    X_dev = np.concatenate([X_train, X_val], axis=0)
    Y_dev = np.concatenate([Y_train, Y_val], axis=0)

    train_idxs = np.arange(len(X_train))
    val_idxs = np.arange(len(X_train), len(X_dev))
    cv = [(train_idxs, val_idxs)]

    cv_model = ElasticNetCV(l1_ratio=l1_ratio, max_iter=max_iter, cv=cv)
    cv_model.fit(X_dev, Y_dev)

    model = ElasticNet(l1_ratio=l1_ratio, alpha=cv_model.alpha_, max_iter=max_iter)
    model.fit(X_train, Y_train)
    return model


def main():
    args = parse_args()

    counts_df, sample_splits_df = load_counts_and_sample_splits(args)
    preds_df = pd.DataFrame(
        np.nan,
        index=sample_splits_df.index,
        columns=sample_splits_df.columns,
        dtype=float,
    )
    coefs = defaultdict(list)

    for gene in tqdm(counts_df.index):
        # Load gene data
        train_data, val_data, test_data, variants = load_gene_data(
            gene, counts_df, sample_splits_df, args.context_size
        )

        if train_data["X"].shape[1] == 0:
            preds_df.loc[gene, test_data["samples"]] = np.mean(train_data["Y"])
            continue

        scaler = StandardScaler()
        train_data["X"] = scaler.fit_transform(train_data["X"])
        val_data["X"] = scaler.transform(val_data["X"])
        test_data["X"] = scaler.transform(test_data["X"])

        model = train_elastic_net_model(
            train_data["X"], train_data["Y"], val_data["X"], val_data["Y"]
        )
        preds_df.loc[gene, test_data["samples"]] = model.predict(test_data["X"])

        assert len(variants) == len(model.coef_)
        coefs["variant"].extend(variants)
        coefs["gene"].extend([gene] * len(variants))
        coefs["beta"].extend(model.coef_)

    os.makedirs(args.output_dir, exist_ok=True)
    preds_df.to_csv(os.path.join(args.output_dir, "preds.csv"))
    coefs_df = pd.DataFrame(coefs)
    coefs_df.to_csv(os.path.join(args.output_dir, "coefs.csv"))


if __name__ == "__main__":
    main()
