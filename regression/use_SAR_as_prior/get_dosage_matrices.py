import os
import sys
from collections import namedtuple

import pandas as pd
from joblib import Parallel, delayed

sys.path.append("..")
import dosage_utils

GEUVADIS_TPM_PATH = "/data/yosef3/users/ruchir/finetuning-enformer/process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz"
OUTPUT_DIR = "dosage_matrices_maf=5e-2"
SEQLEN = 196_608
MAF = 5e-2

TSS = namedtuple("TSS", ["chrom", "pos"])


def get_gene_to_tss_dict() -> dict[str, TSS]:
    tpm_df = pd.read_csv(GEUVADIS_TPM_PATH, index_col=0)  # [genes, samples]
    tpm_df = tpm_df.reset_index().set_index("our_gene_name")
    tpm_df = tpm_df.loc[tpm_df.index.dropna()]
    assert tpm_df.shape[0] == 3259

    return {
        gene: TSS(chrom, pos)
        for gene, chrom, pos in zip(tpm_df.index, tpm_df.Chr, tpm_df.Coord)
    }


def publish_dosage_matrix(gene: str, tss: TSS):
    genotype_matrix = dosage_utils.get_genotype_matrix(tss.chrom, tss.pos, SEQLEN)
    dosage_matrix = dosage_utils.convert_to_dosage_matrix(genotype_matrix)
    dosage_matrix = dosage_utils.filter_common(dosage_matrix, genotype_matrix, maf=MAF)

    output_path = os.path.join(OUTPUT_DIR, f"{gene}.csv")
    dosage_matrix.to_csv(output_path)


def main():
    gene_to_tss_dict = get_gene_to_tss_dict()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Parallel(n_jobs=60, verbose=10)(
        delayed(publish_dosage_matrix)(gene, tss)
        for gene, tss in gene_to_tss_dict.items()
    )


if __name__ == "__main__":
    main()
