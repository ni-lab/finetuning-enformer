import os
from argparse import ArgumentParser, BooleanOptionalAction
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
    parser.add_argument(
        "--cv", action=BooleanOptionalAction, default=False, help="Use cross-validation"
    )
    parser.add_argument(
        "--cv_train_only",
        action=BooleanOptionalAction,
        default=True,
        help="Only use train samples (not train+val) for cross-validation",
    )
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
    cv: bool,
    cv_train_only: bool,
):
    """
    Returns
        data_by_split (dict[str, dict[str, np.ndarray]])
        variants (list[Variant])
    """
    data_by_split = defaultdict(dict)

    # Get samples
    sample_splits = sample_splits_df.loc[gene]
    train_samples = sample_splits[sample_splits == "train"].index.tolist()
    val_samples = sample_splits[sample_splits == "val"].index.tolist()
    test_samples = sample_splits[sample_splits == "test"].index.tolist()
    if cv:
        dev_samples = train_samples if cv_train_only else train_samples + val_samples
        data_by_split["dev"]["samples"] = dev_samples
        data_by_split["test"]["samples"] = test_samples
    else:
        data_by_split["train"]["samples"] = train_samples
        data_by_split["val"]["samples"] = val_samples
        data_by_split["test"]["samples"] = test_samples

    # Load genotype matrix
    genotype_mtx = utils.get_genotype_matrix(
        counts_df.loc[gene, "Chr"], counts_df.loc[gene, "Coord"], context_size
    )  # [variants, samples]

    # Filter variants by MAF/HWE and create X matrices
    def __conv_genotype_mtx_to_X(genotype_mtx):
        dosage_mtx = utils.convert_to_dosage_matrix(genotype_mtx)  # [variants, samples]
        return dosage_mtx.to_numpy(dtype=np.float64).T  # [samples, variants]

    if cv:
        dev_genotype_mtx = genotype_mtx[dev_samples]
        dev_genotype_mtx = utils.filter_common(dev_genotype_mtx)
        dev_genotype_mtx = utils.filter_hwe(dev_genotype_mtx)
        test_genotype_mtx = genotype_mtx.loc[dev_genotype_mtx.index, test_samples]

        data_by_split["dev"]["X"] = __conv_genotype_mtx_to_X(dev_genotype_mtx)
        data_by_split["test"]["X"] = __conv_genotype_mtx_to_X(test_genotype_mtx)
    else:
        train_genotype_mtx = genotype_mtx[train_samples]
        train_genotype_mtx = utils.filter_common(train_genotype_mtx)
        train_genotype_mtx = utils.filter_hwe(train_genotype_mtx)
        val_genotype_mtx = genotype_mtx.loc[train_genotype_mtx.index, val_samples]
        test_genotype_mtx = genotype_mtx.loc[train_genotype_mtx.index, test_samples]

        data_by_split["train"]["X"] = __conv_genotype_mtx_to_X(train_genotype_mtx)
        data_by_split["val"]["X"] = __conv_genotype_mtx_to_X(val_genotype_mtx)
        data_by_split["test"]["X"] = __conv_genotype_mtx_to_X(test_genotype_mtx)

    # Create Y matrices
    for split in data_by_split:
        samples = data_by_split[split]["samples"]
        data_by_split[split]["Y"] = counts_df.loc[gene, samples].to_numpy()

    variants = dev_genotype_mtx.index if cv else train_genotype_mtx.index
    return (data_by_split, variants.tolist())


def standardize_X_matrices(
    data_by_split: dict[str, dict[str, np.ndarray]], fitting_split: str
):
    # Fit scaler on dev/train data
    scaler = StandardScaler()
    data_by_split[fitting_split]["X"] = scaler.fit_transform(
        data_by_split[fitting_split]["X"]
    )

    # Transform remaining splits
    for split in filter(lambda x: x != fitting_split, data_by_split):
        data_by_split[split]["X"] = scaler.transform(data_by_split[split]["X"])


@ignore_warnings(category=ConvergenceWarning)
def train_enet_without_cv(
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
    return model, cv_model.alpha_


@ignore_warnings(category=ConvergenceWarning)
def train_enet_with_cv(
    X_dev: np.ndarray,
    Y_dev: np.ndarray,
    l1_ratio: float = 0.5,
    max_iter: int = 1_000,
    cv: int = 10,
    n_jobs: int = -1,
):
    # Determine optimal alpha using ElasticNetCV
    cv_model = ElasticNetCV(l1_ratio=l1_ratio, max_iter=max_iter, cv=cv, n_jobs=n_jobs)
    cv_model.fit(X_dev, Y_dev)

    # Train ElasticNet model using optimal alpha
    model = ElasticNet(l1_ratio=l1_ratio, alpha=cv_model.alpha_, max_iter=max_iter)
    model.fit(X_dev, Y_dev)
    return model, cv_model.alpha_


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
    alphas = defaultdict(list)

    for gene in tqdm(counts_df.index):
        # Load gene data
        data_by_split, variants = load_gene_data(
            gene,
            counts_df,
            sample_splits_df,
            args.context_size,
            args.cv,
            args.cv_train_only,
        )

        # Skip gene if it has no variants
        fitting_split = "dev" if args.cv else "train"
        if data_by_split[fitting_split]["X"].shape[1] == 0:
            preds_df.loc[gene, data_by_split["test"]["samples"]] = np.mean(
                data_by_split[fitting_split]["Y"]
            )
            continue

        # Standardize X matrices, train model, and predict
        standardize_X_matrices(data_by_split, fitting_split)
        if args.cv:
            model, alpha = train_enet_with_cv(
                data_by_split["dev"]["X"],
                data_by_split["dev"]["Y"],
            )
        else:
            model, alpha = train_enet_without_cv(
                data_by_split["train"]["X"],
                data_by_split["train"]["Y"],
                data_by_split["val"]["X"],
                data_by_split["val"]["Y"],
            )
        preds_df.loc[gene, data_by_split["test"]["samples"]] = model.predict(
            data_by_split["test"]["X"]
        )

        assert len(variants) == len(model.coef_)
        coefs["variant"].extend(variants)
        coefs["gene"].extend([gene] * len(variants))
        coefs["beta"].extend(model.coef_)

        alphas["gene"].append(gene)
        alphas["alpha"].append(alpha)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    preds_df.to_csv(os.path.join(args.output_dir, "preds.csv"))

    coefs_df = pd.DataFrame(coefs)
    coefs_df.to_csv(os.path.join(args.output_dir, "coefs.csv"))

    alphas_df = pd.DataFrame(alphas)
    alphas_df = alphas_df.set_index("gene")
    alphas_df.to_csv(os.path.join(args.output_dir, "alphas.csv"))


if __name__ == "__main__":
    main()
