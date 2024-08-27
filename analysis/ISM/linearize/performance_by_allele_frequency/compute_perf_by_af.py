""""Compute performance of linearized Enformer models, filtering variants by allele frequency.

We support computing performance using two different filtering strategies:
(a) Filter variants by a minimum alternate allele frequency threshold.
    - If the variant's frequency is below the threshold, set its dosage to 0.
(b) Filter alleles by a minimum minor allele frequency threshold.
    - If the allele (either reference or alternate) has a frequency below the threshold, we assign
    the dosage to the major allele.
"""

import os
import sys
import warnings
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
from genomic_utils.variant import Variant
from scipy.stats import ConstantInputWarning, pearsonr
from tqdm import tqdm

sys.path.append("../../..")
import evaluation_utils

sys.path.append("../../predixcan_lite/")
import utils


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "finetuned_ism_dir", type=str, help="Directory containing ISM scores"
    )
    parser.add_argument(
        "model_preds",
        type=str,
        help="Path containing model predictions for all samples and genes",
    )
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--seqlen", type=int, default=384 * 128)
    parser.add_argument(
        "--counts_path",
        type=str,
        default="../../../../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz",
    )
    return parser.parse_args()


def load_genotype_and_dosage_matrix(
    gene: str, counts_df: pd.DataFrame, seqlen: int, samples: list[str]
) -> pd.DataFrame:
    genotype_mtx = utils.get_genotype_matrix(
        counts_df.loc[gene, "Chr"], counts_df.loc[gene, "Coord"], seqlen
    )  # (variants, samples)
    genotype_mtx = genotype_mtx.loc[:, samples]

    # Remove variants that have zero allele counts (across all samples)
    dosage_mtx = utils.convert_to_dosage_matrix(genotype_mtx)
    allele_counts = dosage_mtx.sum(axis=1)
    genotype_mtx = genotype_mtx.loc[allele_counts > 0]
    dosage_mtx = dosage_mtx.loc[allele_counts > 0]

    return (genotype_mtx, dosage_mtx)


def load_ism_scores(gene: str, finetuned_ism_dir: str, variants: pd.Index) -> pd.Series:
    ism_scores_path = os.path.join(finetuned_ism_dir, f"{gene}.csv")
    variants_str = [str(v) for v in variants]
    ism_scores = pd.read_csv(ism_scores_path, index_col=0).loc[variants_str, "ISM"]
    ism_scores.index = ism_scores.index.map(Variant.create_from_str)
    assert ism_scores.index.equals(variants)
    return ism_scores


def compute_variant_allele_frequency(
    genotype_mtx: pd.DataFrame, dosage_mtx: pd.DataFrame
) -> pd.Series:
    allele_counts = dosage_mtx.sum(axis=1)
    allele_nums = genotype_mtx.applymap(utils.count_total_alleles).sum(axis=1)
    allele_freqs = allele_counts / allele_nums
    assert allele_freqs.notnull().all()
    assert (allele_freqs >= 0).all() and (allele_freqs <= 1).all()
    return allele_freqs


def compute_linearized_model_preds(
    dosage_mtx: pd.DataFrame, ism_scores: pd.Series
) -> np.ndarray:
    assert dosage_mtx.index.equals(ism_scores.index)
    D = dosage_mtx.to_numpy().T  # (samples, variants)
    S = ism_scores.to_numpy()  # (variants,)
    return D @ S


def compute_pearson_corr(Y, Y_hat):
    corr = pearsonr(Y, Y_hat)[0]
    return corr if not np.isnan(corr) else 0.0


def assert_all_df_rows_same(df: pd.DataFrame):
    assert (df == df.iloc[0]).all().all()


def add_ref_variants(
    genotype_mtx: pd.DataFrame, dosage_mtx: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add fake reference 'variants' to genotype and dosage dataframes"""
    # Map ref variants to alt variants
    ref_to_alt = defaultdict(list)
    for alt in genotype_mtx.index:
        ref = Variant(alt.chrom, alt.pos, alt.ref, alt.ref)
        ref_to_alt[ref].append(alt)

    # Compute allele counts (AC) and allele numbers (AN) for reference variants
    allele_counts = dosage_mtx
    allele_numbers = genotype_mtx.applymap(utils.count_total_alleles)

    ref_to_ac = {}
    ref_to_an = {}
    for ref in ref_to_alt:
        alts = ref_to_alt[ref]
        ref_counts = (allele_numbers.loc[alts] - allele_counts.loc[alts]).sum(
            axis=0
        )  # (samples,)
        assert (ref_counts >= 0).all() and (ref_counts <= 2).all()
        ref_to_ac[ref] = ref_counts

        assert_all_df_rows_same(allele_numbers.loc[alts])
        ref_to_an[ref] = allele_numbers.loc[alts[0]]
        assert (ref_to_an[ref] >= 0).all() and (ref_to_an[ref] <= 2).all()

    # Add ref variants to dosage matrix
    ref_variants = list(ref_to_ac.keys())
    assert set(ref_variants).isdisjoint(dosage_mtx.index)
    ref_dosage_mtx = pd.DataFrame(
        [ref_to_ac[ref] for ref in ref_variants],
        index=ref_variants,
        columns=dosage_mtx.columns,
    )
    dosage_mtx = pd.concat([dosage_mtx, ref_dosage_mtx], axis=0)

    # Add ref variants to genotype matrix
    def __create_genotype(dosage: int, an: int) -> str:
        assert dosage <= an <= 2
        if an == 0:
            return "."
        else:
            return "/".join(["1"] * dosage + ["0"] * (an - dosage))

    ref_an_mtx = pd.DataFrame(
        [ref_to_an[ref] for ref in ref_variants],
        index=ref_variants,
        columns=genotype_mtx.columns,
    )

    ref_gt_mtx = pd.DataFrame(
        np.vectorize(__create_genotype)(
            ref_dosage_mtx.to_numpy(), ref_an_mtx.to_numpy()
        ),
        index=ref_variants,
        columns=genotype_mtx.columns,
    )
    genotype_mtx = pd.concat([genotype_mtx, ref_gt_mtx], axis=0)

    return (genotype_mtx, dosage_mtx, ref_variants)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    preds_df = evaluation_utils.load_finetuned_model_predictions(args.model_preds)
    genes = preds_df.index.tolist()
    samples = preds_df.columns.tolist()

    # Set allele frequency thresholds
    n_alleles = 2 * len(samples)
    thresholds = np.arange(1 / n_alleles, 1, 2 / n_alleles)

    # Compute performance using alternate allele frequency filtering
    perf = np.full((len(genes), len(thresholds)), np.nan)
    for row, gene in tqdm(enumerate(genes)):
        # Load genotype and dosage matrices of size (variants, samples)
        genotype_mtx, dosage_mtx = load_genotype_and_dosage_matrix(
            gene, counts_df, args.seqlen, samples
        )

        # Get ISM scores with the same variant order as the dosage matrix
        ism_scores = load_ism_scores(gene, args.finetuned_ism_dir, dosage_mtx.index)
        assert ism_scores.isnull().sum() == 0

        # Get variant allele frequency
        allele_freqs = compute_variant_allele_frequency(genotype_mtx, dosage_mtx)

        # Get test samples and ground-truth counts
        test_samples = preds_df.loc[gene].dropna().index.tolist()
        test_idxs = [samples.index(s) for s in test_samples]
        assert len(test_samples) == 77 or len(test_samples) == 421
        Y = counts_df.loc[gene, test_samples].to_numpy()

        # Compute performance by filtering variants by alternate allele frequency
        for col, threshold in enumerate(thresholds):
            dosage_mtx_f = dosage_mtx.copy()
            dosage_mtx_f.loc[allele_freqs < threshold] = 0
            Y_hat = compute_linearized_model_preds(dosage_mtx_f, ism_scores)[test_idxs]
            perf[row, col] = compute_pearson_corr(Y, Y_hat)

    perf_df = pd.DataFrame(perf, index=genes, columns=thresholds)
    output_path = os.path.join(
        args.output_dir, "pearson.alternate_allele_thresholds.csv"
    )
    perf_df.to_csv(output_path)

    # Compute performance using minor allele frequency filtering
    perf = np.full((len(genes), len(thresholds)), np.nan)
    for row, gene in tqdm(enumerate(genes)):
        # Load genotype and dosage matrices of size (variants, samples)
        genotype_mtx, dosage_mtx = load_genotype_and_dosage_matrix(
            gene, counts_df, args.seqlen, samples
        )
        alt_variants = dosage_mtx.index.copy()

        if genotype_mtx.shape[0] == 0:
            # np.vectorize in add_ref_variants will fail if there are no variants. Skip gene.
            perf[row] = 0.0
            continue

        # Add fake reference variants to genotype and dosage matrices
        genotype_mtx, dosage_mtx, ref_variants = add_ref_variants(
            genotype_mtx, dosage_mtx
        )

        # Get ISM scores with the same variant order as the dosage matrix. Add ISM scores of 0.0 for
        # the fake reference variants.
        ism_scores = load_ism_scores(gene, args.finetuned_ism_dir, alt_variants)
        ism_scores = pd.concat([ism_scores, pd.Series(0, index=ref_variants)], axis=0)
        assert genotype_mtx.index.equals(ism_scores.index)

        # Get variant allele frequency
        allele_freqs = compute_variant_allele_frequency(genotype_mtx, dosage_mtx)

        # Map each variant to the major variant at the same locus
        allele_freqs_per_locus = defaultdict(dict)
        for v in allele_freqs.index:
            allele_freqs_per_locus[v.to_locus()][v] = allele_freqs[v]

        major_variant_per_locus = {}
        for locus in allele_freqs_per_locus:
            assert sum(allele_freqs_per_locus[locus].values()) == 1.0
            major_variant_per_locus[locus] = max(
                allele_freqs_per_locus[locus], key=allele_freqs_per_locus[locus].get
            )

        major_variant_map = {
            v: major_variant_per_locus[v.to_locus()] for v in allele_freqs.index
        }
        all_major_variants = set(major_variant_map.values())

        # Get test samples and ground-truth counts
        test_samples = preds_df.loc[gene].dropna().index.tolist()
        test_idxs = [samples.index(s) for s in test_samples]
        assert len(test_samples) == 77 or len(test_samples) == 421
        Y = counts_df.loc[gene, test_samples].to_numpy()

        # Compute performance by filtering variants by allele frequency
        for col, threshold in enumerate(thresholds):
            dosage_mtx_f = dosage_mtx.copy()

            # Get rare variants that are not the major variant at their locus
            rare_variants = set(allele_freqs[allele_freqs < threshold].index)
            rare_variants = list(rare_variants - all_major_variants)

            # Get the corresponding major variants
            major_variants = [major_variant_map[v] for v in rare_variants]

            # Set the dosage of rare variants to 0 and add the dosage of the rare variants to the
            # dosage of their corresponding major variants
            dosage_mtx_f.loc[major_variants] += dosage_mtx_f.loc[rare_variants].values
            dosage_mtx_f.loc[rare_variants] = 0.0

            Y_hat = compute_linearized_model_preds(dosage_mtx_f, ism_scores)[test_idxs]
            perf[row, col] = compute_pearson_corr(Y, Y_hat)

    perf_df = pd.DataFrame(perf, index=genes, columns=thresholds)
    output_path = os.path.join(args.output_dir, "pearson.minor_allele_thresholds.csv")
    perf_df.to_csv(output_path)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    main()
