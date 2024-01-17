import operator
import os
from collections import defaultdict
from functools import reduce

import h5py
import numpy as np
import pandas as pd
from genomic_utils.variant import Variant
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, quantile_transform
from tqdm import tqdm

DOSAGE_MATRICES_DIR = "dosage_matrices_maf=5e-2"
ENFORMER_PREDICTIONS_DIR = "/data/yosef3/scratch/ruchir/data/enformer_hg19_predictions"
GEUVADIS_METADATA_PATH = "/data/yosef3/users/ruchir/pgp_uq/data/E-GEUV-1.sdrf.txt"
GEUVADIS_TPM_PATH = "/data/yosef3/users/ruchir/finetuning-enformer/process_geuvadis_data/tpm/tpm_pca_annot.csv.gz"
OUTPUT_DIR = "predictions_track=69,5110_maf=5e-2_c=True"
USE_GROUPED_CV = False

TRACK_IDXS = [69, 5110]
TRACK_WEIGHTS = [0.5, 0.5]


def load_gene_expression_counts() -> pd.DataFrame:
    tpm_df = pd.read_csv(GEUVADIS_TPM_PATH, index_col=0)
    tpm_df = tpm_df.reset_index().set_index("our_gene_name")
    tpm_df = tpm_df.loc[tpm_df.index.dropna()]
    return tpm_df


def load_sar_scores_for_chrom(chrom: int, variants: set) -> dict[str, float]:
    h5_path = os.path.join(
        ENFORMER_PREDICTIONS_DIR, f"1000G.MAF_threshold=0.005.{chrom}.h5"
    )

    sar_scores = {}
    with h5py.File(h5_path, "r") as f:
        chr = f["chr"][:].astype(str)
        pos = f["pos"][:].astype(int)
        ref = f["ref"][:].astype(str)
        alt = f["alt"][:].astype(str)

        scores = np.abs(f["SAR"][:, TRACK_IDXS].astype(float))
        weighted_scores = scores @ np.asarray(TRACK_WEIGHTS)

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


def load_sar_scores() -> dict[str, float]:
    # Get variants for which we need scores from dosage matrices
    variants = set()
    for fname in tqdm(os.listdir(DOSAGE_MATRICES_DIR), desc="Loading variants"):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(DOSAGE_MATRICES_DIR, fname)
        df = pd.read_csv(fpath, header=0, usecols=[0], names=["variant"])
        variants.update(df["variant"].tolist())
    print(f"Loaded {len(variants)} variants from dosage matrices")

    # Load |SAR| scores
    sar_scores = Parallel(n_jobs=23, verbose=10)(
        delayed(load_sar_scores_for_chrom)(chrom, variants) for chrom in range(1, 23)
    )
    sar_scores = reduce(operator.ior, sar_scores, {})

    n_total = len(variants)
    n_with_scores = len(sar_scores)
    print(f"Loaded SAR scores for {n_with_scores}/{n_total} variants")
    return sar_scores


def load_sample_ancestries() -> dict[str, str]:
    metadata_df = pd.read_csv(GEUVADIS_METADATA_PATH, sep="\t", header=0, index_col=0)
    return metadata_df["Characteristics[ancestry category]"].to_dict()


def run_ridge(X_train, X_test, Y_train) -> np.ndarray:
    model = RidgeCV(alphas=np.logspace(-6, 6, 50))
    model.fit(X_train, Y_train)
    return model.predict(X_test)


def run_sklearn_ridge_with_prior(
    X_train,
    X_test,
    Y_train,
    priors,
    ancestries_train,
    cs: list = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    scale_dosages: bool = False,
) -> np.ndarray:
    if scale_dosages:
        # Scale dosages to have mean zero and unit variance for each variant
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    best_c, best_alpha, best_score = None, None, -np.inf
    for c in cs:
        cov = np.diag(priors**c)
        X_train_scaled = X_train @ np.sqrt(cov)

        if USE_GROUPED_CV:
            cv = LeaveOneGroupOut().split(
                X_train_scaled, Y_train, groups=ancestries_train
            )
        else:
            cv = None

        model = RidgeCV(
            alphas=np.logspace(-6, 6, 50), cv=cv, scoring="neg_mean_squared_error"
        )
        model.fit(X_train_scaled, Y_train)
        if model.best_score_ > best_score:
            best_c, best_alpha, best_score = c, model.alpha_, model.best_score_

    # Train model with best c and alpha
    cov = np.diag(priors**best_c)
    X_train = X_train @ np.sqrt(cov)
    X_test = X_test @ np.sqrt(cov)

    model = Ridge(alpha=best_alpha)
    model.fit(X_train, Y_train)
    return model.predict(X_test)


def run_manual_ridge_with_prior(
    X_train, X_test, Y_train, priors, cv_folds: int = 5
) -> np.ndarray:
    cov = np.diag(priors)
    cov_inv = np.linalg.inv(cov)

    def compute_beta(X, Y, cov_inv, alpha):
        return np.linalg.inv(X.T @ X + alpha * cov_inv) @ X.T @ Y

    def compute_mse(X, Y, beta):
        return np.mean((Y - X @ beta) ** 2)

    # Choose alpha by 5-fold cross validation
    alphas = np.logspace(-6, 6, 50)
    cv_errors = np.zeros((alphas.size, cv_folds))
    kf = KFold(n_splits=cv_folds, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]
        for i, alpha in enumerate(alphas):
            beta = compute_beta(X_train_fold, Y_train_fold, cov_inv, alpha)
            cv_errors[i, fold] = compute_mse(X_val_fold, Y_val_fold, beta)
    mean_cv_errors = np.mean(cv_errors, axis=1)
    best_alpha = alphas[np.argmin(mean_cv_errors)]

    # Train model with best alpha
    beta = compute_beta(X_train, Y_train, cov_inv, best_alpha)

    # Make predictions
    return X_test @ beta


def main():
    tpm_df = load_gene_expression_counts()
    sar_scores = load_sar_scores()
    samples_ancestries = load_sample_ancestries()

    ridge_naive_predictions = defaultdict(dict)
    ridge_prior_predictions = defaultdict(dict)
    ground_truth = defaultdict(dict)

    for fname in tqdm(os.listdir(DOSAGE_MATRICES_DIR), desc="Training models"):
        if not fname.endswith(".csv"):
            continue
        gene = fname.rsplit(".", maxsplit=1)[0]
        dosages = pd.read_csv(
            os.path.join(DOSAGE_MATRICES_DIR, fname), index_col=0
        )  # [variants, samples]
        samples = np.intersect1d(dosages.columns, tpm_df.columns)

        X = dosages[samples].to_numpy(dtype=np.float32).T  # [samples, variants]
        Y = tpm_df.loc[gene, samples].to_numpy()  # [samples]
        Y = quantile_transform(
            Y.reshape(-1, 1), n_quantiles=Y.size, output_distribution="normal"
        ).flatten()

        ancestries = np.asarray([samples_ancestries[s] for s in samples])
        is_YRI = ancestries == "Yoruba"
        X_train, Y_train, ancestries_train = X[~is_YRI], Y[~is_YRI], ancestries[~is_YRI]
        X_test, Y_test = X[is_YRI], Y[is_YRI]

        # Train naive ridge regression model
        Y_naive_pred = run_ridge(X_train, X_test, Y_train)

        # Train ridge regression model with |SAR| as prior
        priors = np.asarray([sar_scores.get(v, np.nan) for v in dosages.index])
        priors[priors == 0] = np.min(priors[priors != 0])
        if np.all(np.isnan(priors)):
            priors = np.ones_like(priors)
        elif np.any(np.isnan(priors)):
            priors = np.nan_to_num(priors, nan=np.nanmean(priors))
        assert np.all(priors > 0)

        Y_prior_pred = run_sklearn_ridge_with_prior(
            X_train, X_test, Y_train, priors, ancestries_train
        )

        # Save predictions
        ridge_naive_predictions[gene] = {
            s: Y_naive_pred[i] for i, s in enumerate(samples[is_YRI])
        }
        ridge_prior_predictions[gene] = {
            s: Y_prior_pred[i] for i, s in enumerate(samples[is_YRI])
        }
        ground_truth[gene] = {s: Y_test[i] for i, s in enumerate(samples[is_YRI])}

    # Create dataframe of predictions and ground truth with genes as columns and samples as rows
    ridge_naive_predictions_df = pd.DataFrame(ridge_naive_predictions)
    ridge_prior_predictions_df = pd.DataFrame(ridge_prior_predictions)
    ground_truth_df = pd.DataFrame(ground_truth)

    # Save predictions and ground truth
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ridge_naive_predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, "ridge_naive_predictions.csv")
    )
    ridge_prior_predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, "ridge_prior_predictions.csv")
    )
    ground_truth_df.to_csv(os.path.join(OUTPUT_DIR, "ground_truth.csv"))


if __name__ == "__main__":
    main()
