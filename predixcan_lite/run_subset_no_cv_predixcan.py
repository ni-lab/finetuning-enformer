# Training PrediXcan models on a subset of the samples, without using cross-validation.
# We repeat this process n times using a different subset of samples each time.

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import utils
from run_no_cv_predixcan import train_elastic_net_model
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("subset_fraction", type=float)
    parser.add_argument("--n_replicates", type=int, default=3)
    parser.add_argument("--context_size", type=int, default=1_000_000)
    parser.add_argument("--random_seed", type=int, default=42)
    # fmt: off
    parser.add_argument("--counts_path", type=str, default="../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--sample_splits_path", type=str, default="h5_bins_384_chrom_split/sample_splits.h5_bins_384_chrom_split.csv")
    # fmt: on
    return parser.parse_args()


def load_and_split_genotype_mtx(
    gene: str,
    counts_df: pd.DataFrame,
    sample_splits_df: pd.DataFrame,
    context_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
        train_genotype_mtx (pd.DataFrame): [variants, samples]
        val_genotype_mtx (pd.DataFrame): [variants, samples]
        test_genotype_mtx (pd.DataFrame): [variants, samples]
    """
    sample_splits = sample_splits_df.loc[gene]
    train_samples = sample_splits[sample_splits == "train"].index.tolist()
    val_samples = sample_splits[sample_splits == "val"].index.tolist()
    test_samples = sample_splits[sample_splits == "test"].index.tolist()

    genotype_mtx = utils.get_genotype_matrix(
        counts_df.loc[gene, "Chr"], counts_df.loc[gene, "Coord"], context_size
    )  # [variants, samples]
    train_genotype_mtx = genotype_mtx[train_samples]
    val_genotype_mtx = genotype_mtx[val_samples]
    test_genotype_mtx = genotype_mtx[test_samples]
    return (train_genotype_mtx, val_genotype_mtx, test_genotype_mtx)


def convert_geno_mtx_to_dosage_arr(geno_mtx: pd.DataFrame) -> np.ndarray:
    return utils.convert_to_dosage_matrix(geno_mtx).to_numpy(dtype=np.float64).T


def scale_dosage_arrs(
    train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)
    return (train_X, val_X, test_X)


def main():
    args = parse_args()
    sample_splits_df = pd.read_csv(args.sample_splits_path, index_col=0)
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    preds_dfs = [
        pd.DataFrame(
            np.nan, index=counts_df.index, columns=sample_splits_df.columns, dtype=float
        )
        for _ in range(args.n_replicates)
    ]  # list of DataFrames containing predictions for each replicate

    rng = np.random.default_rng(seed=args.random_seed)

    for gene in tqdm(sorted(sample_splits_df.index)):
        (
            orig_train_geno_mtx,
            orig_val_geno_mtx,
            orig_test_geno_mtx,
        ) = load_and_split_genotype_mtx(
            gene, counts_df, sample_splits_df, args.context_size
        )

        n_train_subset = int(args.subset_fraction * orig_train_geno_mtx.shape[1])
        for i in range(args.n_replicates):
            # Randomly sample a subset of train samples
            train_samples_subset = rng.choice(
                orig_train_geno_mtx.columns, n_train_subset, replace=False
            )

            # Filter genotype matrices by the subset of train samples
            my_train_geno_mtx = orig_train_geno_mtx[train_samples_subset]
            my_train_geno_mtx = utils.filter_common(my_train_geno_mtx)
            my_train_geno_mtx = utils.filter_hwe(my_train_geno_mtx)
            my_val_geno_mtx = orig_val_geno_mtx.loc[my_train_geno_mtx.index]
            my_test_geno_mtx = orig_test_geno_mtx.loc[my_train_geno_mtx.index]

            # Convert genotype matrices to dosage arrays
            train_X = convert_geno_mtx_to_dosage_arr(my_train_geno_mtx)
            val_X = convert_geno_mtx_to_dosage_arr(my_val_geno_mtx)
            test_X = convert_geno_mtx_to_dosage_arr(my_test_geno_mtx)

            # Get Y arrays
            train_Y = counts_df.loc[gene, train_samples_subset].to_numpy()
            val_Y = counts_df.loc[gene, my_val_geno_mtx.columns].to_numpy()

            # Scale dosage arrays if they are not empty
            if train_X.shape[1] == 0:
                preds_dfs[i].loc[gene, my_test_geno_mtx.columns] = np.mean(train_Y)
                continue

            train_X, val_X, test_X = scale_dosage_arrs(train_X, val_X, test_X)

            # Train elastic net model
            model = train_elastic_net_model(train_X, train_Y, val_X, val_Y)
            preds_dfs[i].loc[gene, my_test_geno_mtx.columns] = model.predict(test_X)

    for i, preds_df in enumerate(preds_dfs):
        output_fname = (
            f"subset_fraction_{args.subset_fraction:.1f}_replicate_{i + 1}.csv"
        )
        preds_df.to_csv(os.path.join(args.output_dir, output_fname))


if __name__ == "__main__":
    main()
