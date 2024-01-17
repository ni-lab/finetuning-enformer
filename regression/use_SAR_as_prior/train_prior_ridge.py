import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
import utils
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument(
        "--counts_path",
        type=str,
        default="../../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz",
    )
    parser.add_argument(
        "--dosage_matrices_dir", type=str, default="dosage_matrices_maf=5e-2"
    )
    return parser.parse_args()


def compute_priors(sar_scores: dict[str, float], variants: list[str]) -> np.ndarray:
    priors = np.asarray([sar_scores.get(v, np.nan) for v in variants])
    priors[priors == 0] = np.min(priors[priors != 0])
    if np.all(np.isnan(priors)):
        priors = np.ones_like(priors)
    elif np.any(np.isnan(priors)):
        priors = np.nan_to_num(priors, nan=np.nanmean(priors))
    assert np.all(priors > 0)
    return priors


def train_and_predict_ridge(
    X_train, X_test, Y_train, priors, alphas=np.logspace(-6, 6, 50)
):
    X_train = X_train @ np.sqrt(np.diag(priors))
    X_test = X_test @ np.sqrt(np.diag(priors))

    model = RidgeCV(alphas=alphas)
    model.fit(X_train, Y_train)
    return model.predict(X_test)


def main():
    args = parse_args()
    counts_df = utils.load_gene_expression_counts(args.counts_path)
    sample_ancestries = utils.load_sample_ancestries()
    sar_scores = utils.load_sar_scores(args.dosage_matrices_dir)

    # We have two tasks for each gene:
    # 1. Train on random subset of samples and predict on remaining samples
    # 2. Train on non-YRI samples and predict on YRI samples

    all_predictions = {}  # gene -> pd.Series
    all_ground_truth = {}  # gene -> pd.Series

    yri_predictions = {}  # gene -> pd.Series
    yri_ground_truth = {}  # gene -> pd.Series

    for fname in tqdm(os.listdir(args.dosage_matrices_dir), desc="Training models"):
        if not fname.endswith(".csv"):
            continue
        gene = fname.rsplit(".", maxsplit=1)[0]
        dosages = pd.read_csv(
            os.path.join(args.dosage_matrices_dir, fname), index_col=0
        )  # [variants, samples]
        samples = np.intersect1d(dosages.columns, counts_df.columns)

        X = dosages[samples].to_numpy(dtype=np.float32).T  # [samples, variants]
        Y = counts_df.loc[gene, samples].to_numpy()  # [samples]
        priors = compute_priors(sar_scores, dosages.index.tolist())

        # 1. Train on random subset of samples and predict on remaining samples through 10-fold CV
        Y_all_pred = np.zeros_like(Y)
        for train_idx, test_idx in KFold(n_splits=10, shuffle=True).split(X):
            Y_all_pred[test_idx] = train_and_predict_ridge(
                X[train_idx], X[test_idx], Y[train_idx], priors
            )
        all_predictions[gene] = pd.Series(Y_all_pred, index=samples)
        all_ground_truth[gene] = pd.Series(Y, index=samples)

        # 2. Train on non-YRI samples and predict on YRI samples
        ancestries = np.asarray([sample_ancestries[s] for s in samples])
        train = ancestries != "Yoruba"
        Y_yri_pred = train_and_predict_ridge(X[train], X[~train], Y[train], priors)
        yri_predictions[gene] = pd.Series(Y_yri_pred, index=samples[~train])
        yri_ground_truth[gene] = pd.Series(Y[~train], index=samples[~train])

    # Save predictions and ground truth
    all_predictions_df = pd.DataFrame(all_predictions)
    all_ground_truth_df = pd.DataFrame(all_ground_truth)
    yri_predictions_df = pd.DataFrame(yri_predictions)
    yri_ground_truth_df = pd.DataFrame(yri_ground_truth)

    os.makedirs(args.output_dir, exist_ok=True)
    all_predictions_df.to_csv(os.path.join(args.output_dir, "all_predictions.csv"))
    all_ground_truth_df.to_csv(os.path.join(args.output_dir, "all_ground_truth.csv"))
    yri_predictions_df.to_csv(os.path.join(args.output_dir, "yri_predictions.csv"))
    yri_ground_truth_df.to_csv(os.path.join(args.output_dir, "yri_ground_truth.csv"))


if __name__ == "__main__":
    main()
