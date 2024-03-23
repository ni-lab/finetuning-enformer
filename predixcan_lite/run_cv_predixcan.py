from argparse import ArgumentParser

import numpy as np
import pandas as pd
import utils
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_preds_path", type=str)
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
        X_dev (np.ndarray): [n_samples, n_variants]
        X_test (np.ndarray): [n_samples, n_variants]
        Y_dev (np.ndarray): [n_samples]
        Y_test (np.ndarray): [n_samples]
        dev_samples (list[str])
        test_samples (list[str])
        variants (list[str])
    """

    # Get train, val, and test samples
    sample_splits = sample_splits_df.loc[gene]
    train_samples = sample_splits[sample_splits == "train"].index.tolist()
    val_samples = sample_splits[sample_splits == "val"].index.tolist()
    test_samples = sample_splits[sample_splits == "test"].index.tolist()
    dev_samples = list(set(train_samples) | set(val_samples))
    assert len(dev_samples) + len(test_samples) == len(sample_splits_df.columns)

    # Load genotype matrix and filter variants by MAF/ HWE
    genotype_mtx = utils.get_genotype_matrix(
        counts_df.loc[gene, "Chr"], counts_df.loc[gene, "Coord"], context_size
    )  # [variants, samples]

    dev_genotype_mtx = genotype_mtx[dev_samples]
    dev_genotype_mtx = utils.filter_common(dev_genotype_mtx)
    dev_genotype_mtx = utils.filter_hwe(dev_genotype_mtx)
    test_genotype_mtx = genotype_mtx.loc[dev_genotype_mtx.index, test_samples]

    # Create X_dev, X_test, Y_dev, and Y_test
    dev_dosage_mtx = utils.convert_to_dosage_matrix(dev_genotype_mtx)
    test_dosage_mtx = utils.convert_to_dosage_matrix(test_genotype_mtx)

    X_dev = dev_dosage_mtx.to_numpy(dtype=np.float64).T
    X_test = test_dosage_mtx.to_numpy(dtype=np.float64).T
    Y_dev = counts_df.loc[gene, dev_samples].to_numpy()
    Y_test = counts_df.loc[gene, test_samples].to_numpy()

    return (
        X_dev,
        X_test,
        Y_dev,
        Y_test,
        dev_samples,
        test_samples,
        dev_dosage_mtx.index,
    )


@ignore_warnings(category=ConvergenceWarning)
def train_elastic_net_model(
    X_dev,
    Y_dev,
    l1_ratio: float = 0.5,
    max_iter: int = 1_000,
    cv: int = 10,
    n_jobs: int = -1,
):
    cv_model = ElasticNetCV(l1_ratio=l1_ratio, max_iter=max_iter, cv=cv, n_jobs=n_jobs)
    cv_model.fit(X_dev, Y_dev)

    model = ElasticNet(l1_ratio=l1_ratio, alpha=cv_model.alpha_, max_iter=max_iter)
    model.fit(X_dev, Y_dev)
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

    for gene in tqdm(counts_df.index):
        # Load gene data
        (
            X_dev,
            X_test,
            Y_dev,
            Y_test,
            dev_samples,
            test_samples,
            variants,
        ) = load_gene_data(gene, counts_df, sample_splits_df, args.context_size)

        scaler = StandardScaler()
        X_dev = scaler.fit_transform(X_dev)
        X_test = scaler.transform(X_test)

        model = train_elastic_net_model(X_dev, Y_dev)
        preds_df.loc[gene, test_samples] = model.predict(X_test)

    preds_df.to_csv(args.output_preds_path)


if __name__ == "__main__":
    main()
