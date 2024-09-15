import warnings
from typing import Optional

import numpy as np
import pandas as pd
from genomic_utils.variant import Variant
from scipy.stats import ConstantInputWarning, linregress, pearsonr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConstantInputWarning)


def compute_drivers(
    dosages_df: pd.DataFrame,
    weights: pd.Series,
    test_samples: list[str],
    test_y_pred: np.ndarray,
    standardize_dosages: bool = False,
    train_samples: Optional[list[str]] = None,
    check_linear_pred_equals_test_y_pred: bool = True,
) -> list[Variant]:
    """Computes driver variants that significantly contribute to model predictions on the test set
    using a forward selection algorithm.

    Note that test_y_preds does not have to equal dosages @  weights for this function to work.
    weights should hold the betas for linear models or ISMs for deep learning models.

    Args:
        dosages_df (pd.DataFrame): DataFrame of size [n_variants, n_samples]
        weights (pd.Series): Series of size [n_variants,]
        standardize_dosages (bool): Whether to standardize dosages
    Returns:
        list of driver variants, sorted by importance
    """
    assert dosages_df.index.equals(weights.index)
    assert len(test_samples) == test_y_pred.shape[0]

    if len(weights) == 0:
        print("No weights found. Skipping driver computation.")
        return []

    # Subset dosages to test samples
    test_dosages_df = dosages_df[test_samples]

    # Standardize dosages to match the prediction pipeline of FUSION models
    if standardize_dosages:
        assert train_samples is not None  # needed to standardize dosages
        train_dosages_df = dosages_df[train_samples]
        scaler = StandardScaler().fit(train_dosages_df.values.T)
        test_dosages_df = pd.DataFrame(
            scaler.transform(test_dosages_df.values.T).T,
            index=test_dosages_df.index,
            columns=test_dosages_df.columns,
        )

    # Get X and w arrays
    X = test_dosages_df.values.T  # [n_test_samples, n_variants]
    w = weights.values  # [n_variants,]

    if check_linear_pred_equals_test_y_pred:
        linear_preds = X @ w
        residual = test_y_pred - linear_preds
        assert np.allclose(residual, residual[0])

    # Compute drivers using the following algorithm:
    #   Sort variants by absolute value of their weight (beta/ISM)
    #   For each driver:
    #     1. Determine if adding a variant to the partial model increases the correlation
    #        with test_y_pred by >= 0.05 and if the variant has a marginal correlation with
    #        test_y_pred (rho > 0 and p_{Bonferroni} < 0.01)
    #     2. If so, add the variant to the list of drivers
    driver_idxs = []
    best_partial_corr = 0.0

    def _compute_partial_model_corr(X: np.ndarray, idxs: list[int]) -> float:
        """Returns the Pearson correlation between the partial model (defined by idxs) and y."""
        X_partial = X[:, idxs]
        w_partial = w[idxs]
        return pearsonr(X_partial @ w_partial, test_y_pred)[0]

    def compute_marginal_corr(X: np.ndarray, idx: int) -> float:
        """Returns the marginal correlation and Bonferroni-corrected p-value of the variant at idx with y."""
        rho, p = pearsonr(X[:, idx] * w[idx], test_y_pred)
        p_adj = p * X.shape[1]
        return rho, p_adj

    sorted_variant_idxs = np.abs(w).argsort()[::-1]
    for variant_idx in sorted_variant_idxs:
        partial_corr = _compute_partial_model_corr(X, driver_idxs + [variant_idx])
        if np.isnan(partial_corr) or (partial_corr < best_partial_corr + 0.05):
            continue

        marginal_corr, p_adj = compute_marginal_corr(X, variant_idx)
        if np.isnan(marginal_corr) or (marginal_corr < 0) or (p_adj > 0.01):
            continue

        driver_idxs.append(variant_idx)
        best_partial_corr = partial_corr

    return weights.index[driver_idxs].tolist()
