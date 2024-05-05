import os
from argparse import ArgumentParser, BooleanOptionalAction

import h5py
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument(
        "--meta_feature_weight_fn", type=str, choices=["uniform", "LCL"], default="LCL"
    )
    parser.add_argument("--use_Z", action=BooleanOptionalAction, default=False)
    return parser.parse_args()


def compute_priors(F: np.ndarray, meta_feature_weight_fn: str):
    """
    Parameters:
        F: np.ndarray of shape (n_variants, n_meta_features)
    Returns:
        priors: np.ndarray of shape (n_variants,)
    """
    if meta_feature_weight_fn == "uniform":
        W = np.ones((F.shape[1],)) / F.shape[1]
    elif meta_feature_weight_fn == "LCL":
        W = np.zeros((F.shape[1],))
        W[69] = 0.5
        W[5110] = 0.5
    else:
        raise ValueError(f"Invalid meta_feature_weight_fn: {meta_feature_weight_fn}")
    return F @ W


def load_data_for_all_genes(dataset_dir: str, use_Z: bool, meta_feature_weight_fn: str):
    train_h5_path = os.path.join(dataset_dir, "train.h5")
    val_h5_path = os.path.join(dataset_dir, "val.h5")
    test_h5_path = os.path.join(dataset_dir, "test.h5")

    def collate_data(h5_path: str, load_priors: bool = True):
        data = {}
        with h5py.File(h5_path, "r") as f:
            genes = f["genes"][:].astype(str)
            data["X"] = {
                g: f[g]["dosages"][:].astype(np.float32)
                for g in tqdm(genes, desc="Loading dosages")
            }
            if load_priors:
                meta_features = {
                    g: f[g]["meta_features"][:].astype(np.float32)
                    for g in tqdm(genes, desc="Loading meta features")
                }
                data["priors"] = {
                    g: compute_priors(meta_features[g], meta_feature_weight_fn)
                    for g in tqdm(genes, desc="Computing priors")
                }
            if use_Z:
                data["Y"] = {g: f[g]["z_scores"][:].astype(np.float32) for g in genes}
            else:
                data["Y"] = {g: f[g]["Y"][:].astype(np.float32) for g in genes}
            data["samples"] = {g: f[g]["samples"][:].astype(str) for g in genes}
        return data

    train_data = collate_data(train_h5_path)
    val_data = collate_data(val_h5_path, load_priors=False)
    test_data = collate_data(test_h5_path, load_priors=False)
    return (train_data, val_data, test_data)


@ignore_warnings(category=ConvergenceWarning)
def train_weighted_ridge(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    priors: np.ndarray,
    alphas=np.logspace(-6, 6, 50),
):
    X_train = X_train @ np.sqrt(
        np.diag(priors)
    )  # reformulate X_train to account for priors
    X_val = X_val @ np.sqrt(np.diag(priors))  # reformulate X_val to account for priors

    X_dev = np.vstack((X_train, X_val))  # (n_train_samples + n_val_samples, n_variants)
    Y_dev = np.concatenate((Y_train, Y_val))  # (n_train_samples + n_val_samples)

    train_idxs = np.arange(X_train.shape[0])
    val_idxs = np.arange(X_train.shape[0], X_dev.shape[0])
    cv = [(train_idxs, val_idxs)]

    # Determine the best alpha using RidgeCV
    cv_model = RidgeCV(alphas=alphas, cv=cv)
    cv_model.fit(X_dev, Y_dev)

    # Train the final model using the best alpha
    model = Ridge(alpha=cv_model.alpha_)
    model.fit(X_train, Y_train)
    return model


def test_weighted_ridge(model, X_test: np.ndarray, priors: np.ndarray):
    X_test = X_test @ np.sqrt(np.diag(priors))
    return model.predict(X_test)


def main():
    args = parse_args()

    train_data, val_data, test_data = load_data_for_all_genes(
        args.dataset_dir, args.use_Z, args.meta_feature_weight_fn
    )
    genes = sorted(train_data["X"].keys())
    assert len(genes) == 3259

    train_samples = set.union(
        *(set(samples) for samples in train_data["samples"].values())
    )
    val_samples = set.union(*(set(samples) for samples in val_data["samples"].values()))
    test_samples = set.union(
        *(set(samples) for samples in test_data["samples"].values())
    )
    samples = sorted(train_samples | val_samples | test_samples)
    sample_to_idx = {sample: idx for idx, sample in enumerate(samples)}
    assert len(samples) == 421

    Y_mtx = np.full((len(genes), len(samples)), np.nan)
    Y_hat_mtx = np.full((len(genes), len(samples)), np.nan)
    for gene_idx, g in tqdm(enumerate(genes)):
        X_train, X_val, X_test = train_data["X"][g], val_data["X"][g], test_data["X"][g]
        Y_train, Y_val, Y_test = train_data["Y"][g], val_data["Y"][g], test_data["Y"][g]
        priors = train_data["priors"][g]
        test_samples = test_data["samples"][g]
        test_sample_idxs = [sample_to_idx[s] for s in test_samples]

        if X_train.shape[1] == 0:
            # No variants for this gene
            Y_mtx[gene_idx, test_sample_idxs] = Y_test
            Y_hat_mtx[gene_idx, test_sample_idxs] = np.mean(Y_train)
            continue

        model = train_weighted_ridge(X_train, Y_train, X_val, Y_val, priors)
        Y_hat_test = test_weighted_ridge(model, X_test, priors)

        Y_mtx[gene_idx, test_sample_idxs] = Y_test
        Y_hat_mtx[gene_idx, test_sample_idxs] = Y_hat_test

    Y_df = pd.DataFrame(Y_mtx, index=genes, columns=samples)
    Y_hat_df = pd.DataFrame(Y_hat_mtx, index=genes, columns=samples)

    os.makedirs(args.output_dir, exist_ok=True)
    Y_df.to_csv(os.path.join(args.output_dir, "counts.csv"))
    Y_hat_df.to_csv(os.path.join(args.output_dir, "preds.csv"))


if __name__ == "__main__":
    main()
