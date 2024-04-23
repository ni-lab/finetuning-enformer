import os
import sys
from argparse import ArgumentParser
from functools import partial

import pandas as pd
from joblib import Parallel, delayed

sys.path.append("/data/yosef3/users/ruchir/finetuning-enformer/predixcan_lite")
import utils


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--context_size", type=int, default=1_000_000)
    parser.add_argument("--n_jobs", type=int, default=30)
    # fmt: off
    parser.add_argument("--counts_path", type=str, default="../../../process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
    parser.add_argument("--splits_path", type=str, default="../splits/random_split.csv")
    # fmt: on
    return parser.parse_args()


def get_genotype_matrix(
    gene: str,
    counts_df: pd.DataFrame,
    context_size: int,
    samples: list[str],
    output_dir: str,
):
    genotype_mtx = utils.get_genotype_matrix(
        counts_df.loc[gene, "Chr"], counts_df.loc[gene, "Coord"], context_size
    )  # (variants, samples)

    # Remove variants that are not present in any of our samples
    genotype_mtx = genotype_mtx.loc[:, samples]
    allele_counts = utils.convert_to_dosage_matrix(genotype_mtx).sum(axis=1)
    genotype_mtx = genotype_mtx.loc[allele_counts > 0]

    output_path = os.path.join(output_dir, f"{gene}.csv")
    genotype_mtx.to_csv(output_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    counts_df = pd.read_csv(args.counts_path, index_col="our_gene_name")
    splits_df = pd.read_csv(args.splits_path, index_col=0)  # (genes, samples)
    genes = splits_df.index.tolist()
    samples = splits_df.columns.tolist()

    get_genotype_matrix_partial = partial(
        get_genotype_matrix,
        counts_df=counts_df,
        context_size=args.context_size,
        samples=samples,
        output_dir=args.output_dir,
    )

    Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(get_genotype_matrix_partial)(g) for g in genes
    )


if __name__ == "__main__":
    main()
