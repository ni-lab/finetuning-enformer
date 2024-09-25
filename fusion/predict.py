"""Compute performance metrics on the test set for all models run through the FUSION pipeline."""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from genomic_utils.variant import Variant
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append("../vcf_utils")
import utils


# fmt: off
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--gene_metadata_path", type=str, required=True, help="Path containing list of genes.")
    parser.add_argument("--weight_dir", type=str, required=True, help="Directory containing model weights.")
    parser.add_argument("--temp_dir", type=str, required=True, help="Directory containing train/test pheno files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions for each model.")
    parser.add_argument("--counts_path", type=str, default="../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--models", nargs="+", type=str, default=["top1", "lasso", "enet", "blup", "bslmm"])
    parser.add_argument("--maf", type=float, default=0.05)
    return parser.parse_args()
# fmt: on


def get_samples_from_pheno_file(pheno_path: str) -> list[str]:
    df = pd.read_csv(
        pheno_path, sep=" ", header=None, names=["sample", "family", "pheno"]
    )
    return df["sample"].tolist()


def get_samples(tmp_dir: str, genes: list[str]) -> list[str]:
    samples = set()
    for g in genes:
        train_pheno_path = os.path.join(tmp_dir, f"{g}.train.pheno")
        test_pheno_path = os.path.join(tmp_dir, f"{g}.test.pheno")
        samples.update(get_samples_from_pheno_file(train_pheno_path))
        samples.update(get_samples_from_pheno_file(test_pheno_path))
    return sorted(list(samples))


def compute_alt_allele_freq(genotype_mtx: pd.DataFrame) -> pd.Series:
    ac = genotype_mtx.applymap(utils.convert_to_dosage).sum(axis=1)
    an = genotype_mtx.applymap(utils.count_total_alleles).sum(axis=1)
    return ac / an


def load_train_and_test_dosages(
    counts_df: pd.DataFrame,
    gene: str,
    train_samples: list[str],
    test_samples: list[str],
    maf: float,
    context_size: int = 128 * 384,
    scale_dosages: bool = True,
    flip_variants: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load genotypes: [n_variants, n_samples]
    genotype_mtx = utils.get_genotype_matrix(
        counts_df.loc[gene, "Chr"], counts_df.loc[gene, "Coord"], context_size, remove_ac0=False
    )

    # Compute alternate allele frequency (AAF)
    train_aaf = compute_alt_allele_freq(genotype_mtx[train_samples])

    # Flip the genotypes of variants with AAF > 0.5.
    # This was done originally because PLINK was outputting major allele/minor allele instead of
    # reference/alternate allele. Now that we've added --keep-allele-order flags this is not needed.
    if flip_variants:

        def _swap_alleles(gt: str) -> str:
            return gt.replace("0", "tmp").replace("1", "0").replace("tmp", "1")

        # Swap the ref/alt alleles of variants with AAF > 0.5
        genotype_mtx.loc[train_aaf > 0.5] = genotype_mtx.loc[train_aaf > 0.5].applymap(
            _swap_alleles
        )

        # Confirm that swap was correct by checking that the new AAF = min(old AAF, 1 - old AAF)
        new_train_aaf = compute_alt_allele_freq(genotype_mtx[train_samples])
        assert np.allclose(
            new_train_aaf.values, np.minimum(train_aaf.values, 1 - train_aaf.values)
        )
        train_aaf = new_train_aaf

        # Switch the ref/alt order of variants with AAF > 0.5
        genotype_mtx.index = genotype_mtx.index.map(
            lambda v: v.switch_ref_alt() if train_aaf.loc[v] > 0.5 else v
        )

    # Filter on AAF
    variants_f = train_aaf[(train_aaf >= maf) & (train_aaf <= 1 - maf)].index
    train_genotype_mtx = genotype_mtx.loc[variants_f, train_samples]
    test_genotype_mtx = genotype_mtx.loc[variants_f, test_samples]

    # Convert to dosage
    train_dosage_mtx = train_genotype_mtx.applymap(utils.convert_to_dosage)
    test_dosage_mtx = test_genotype_mtx.applymap(utils.convert_to_dosage)

    # Scale dosages (i.e. scale rows/variants to have mean 0 and variance 1)
    if scale_dosages and train_dosage_mtx.shape[0] > 0:
        scaler = StandardScaler()
        train_dosage_mtx = pd.DataFrame(
            scaler.fit_transform(train_dosage_mtx.values.T).T,
            index=train_dosage_mtx.index,
            columns=train_dosage_mtx.columns,
        )
        test_dosage_mtx = pd.DataFrame(
            scaler.transform(test_dosage_mtx.values.T).T,
            index=test_dosage_mtx.index,
            columns=test_dosage_mtx.columns,
        )

    return train_dosage_mtx, test_dosage_mtx


def get_expr_from_pheno_file(pheno_path: str) -> np.ndarray:
    df = pd.read_csv(
        pheno_path, sep=" ", header=None, names=["sample", "family", "pheno"]
    )
    return df["pheno"].values


def load_weights(weights_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        weights_path, sep="\t", header=None, names=["id", "alt", "ref", "weight"]
    )
    if df.isnull().values.any():
        print(f"WARNING: NaNs detected in {weights_path}")
        df = df.dropna()

    df["chrom"] = df["id"].apply(lambda x: x.split("_")[1])
    df["pos"] = df["id"].apply(lambda x: int(x.split("_")[2]))

    # plink2R, which is used by FUSION, uses read.table to read in the plink BIM file.
    # This sometimes results in a T allele being read in as a TRUE value. We handle that bug here.
    df["ref"] = df["ref"].replace({True: "T"})
    df["alt"] = df["alt"].replace({True: "T"})

    df["variant"] = [
        Variant(chrom, pos, ref, alt)
        for (chrom, pos, ref, alt) in zip(df["chrom"], df["pos"], df["ref"], df["alt"])
    ]
    df = df.set_index("variant")
    df = df[["weight"]]
    return df


def make_predictions(
    dosage_mtx: pd.DataFrame,  # [variant, sample]
    weights_df: pd.DataFrame,  # [variant],
    intercept: float,
    model: str,
) -> np.ndarray:
    if model == "blup":
        assert set(dosage_mtx.index) == set(weights_df.index)
    else:
        assert set(weights_df.index).issubset(set(dosage_mtx.index))

    weights_df["weight"] = pd.to_numeric(weights_df["weight"], errors="coerce")
    weights_df = weights_df.dropna(subset=["weight"])
    dosage_mtx = dosage_mtx.loc[weights_df.index]

    dosages = dosage_mtx.values.T  # [sample, variant]
    weights = weights_df["weight"].values
    return dosages @ weights + intercept


def main():
    args = parse_args()

    # Get list of genes
    metadata_df = pd.read_csv(args.gene_metadata_path, sep="\t", index_col=0)
    genes = metadata_df.index.tolist()

    # Get list of samples
    samples = get_samples(args.temp_dir, genes)
    assert len(samples) == 421

    # Load counts data
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name").loc[genes]

    # Obtain test set predictions for each model on every gene
    model_preds = {
        model: pd.DataFrame(np.nan, index=genes, columns=samples, dtype=float)
        for model in args.models
    }

    for gene in tqdm(genes):
        train_pheno_path = os.path.join(args.temp_dir, f"{gene}.train.pheno")
        test_pheno_path = os.path.join(args.temp_dir, f"{gene}.test.pheno")
        train_samples = get_samples_from_pheno_file(train_pheno_path)
        test_samples = get_samples_from_pheno_file(test_pheno_path)

        # Load (scaled) test dosages
        _, test_dosages = load_train_and_test_dosages(
            counts_df, gene, train_samples, test_samples, args.maf
        )

        train_Y = get_expr_from_pheno_file(train_pheno_path)

        for model in args.models:
            weights_path = os.path.join(args.weight_dir, f"{gene}.{model}.weights.txt")
            if not os.path.exists(weights_path):
                print(f"WARNING: No weights found for {gene=} ({model=}). Skipping...")
                model_preds[model].loc[gene, test_samples] = np.mean(train_Y)
                continue

            weights = load_weights(weights_path)
            if weights.shape[0] == 0:
                print(f"WARNING: No weights found for {gene=} ({model=}). Skipping...")
                model_preds[model].loc[gene, test_samples] = np.mean(train_Y)
                continue

            test_Y_hat = make_predictions(
                test_dosages, weights, np.mean(train_Y), model
            )

            model_preds[model].loc[gene, test_samples] = test_Y_hat

    # Save performance metrics
    os.makedirs(args.output_dir, exist_ok=True)
    for model in args.models:
        model_preds[model].to_csv(os.path.join(args.output_dir, f"{model}.csv"))


if __name__ == "__main__":
    main()
