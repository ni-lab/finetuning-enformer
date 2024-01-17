import gzip
import os
from argparse import ArgumentParser
from dataclasses import dataclass

import pandas as pd

INPUT_DIR = "/data/yosef3/scratch/ruchir/data/geuvadis/analysis_results"
EXPRESSION_PATH = os.path.join(
    INPUT_DIR, "GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz"
)
EUR_CIS_EQTL_PATH = os.path.join(INPUT_DIR, "EUR373.gene.cis.FDR5.all.rs137.txt.gz")
YRI_CIS_EQTL_PATH = os.path.join(INPUT_DIR, "YRI89.gene.cis.FDR5.all.rs137.txt.gz")
GENCODE_PATH = "/data/yosef3/scratch/ruchir/data/gencode/gencode.v12.annotation.gtf.gz"
OUR_GENES_METADATA_PATH = "/data/yosef3/users/ruchir/pgp_uq/data/eur_eqtl_gene_list.csv"


@dataclass
class Gene:
    def __init__(self, gene_id, gene_name, chrom, tss, strand):
        self.gene_id = gene_id
        self.gene_name = gene_name
        self.chrom = chrom.replace("chr", "")
        self.tss = int(tss)
        self.strand = strand

    def __repr__(self):
        return f"Gene({self.gene_id}, {self.gene_name}, {self.chrom}, {self.tss}, {self.strand})"

    def __hash__(self):
        return hash(self.gene_id)


def get_gtf_attributes_dict(attributes: str) -> dict[str, str]:
    map_ = {}
    for attr in attributes.split(";"):
        attr = attr.strip()
        if attr:
            k, v = attr.split()
            map_[k] = v[1:-1]
    return map_


def load_genes() -> dict[str, Gene]:
    genes_by_id = {}
    for line in gzip.open(GENCODE_PATH, "rt"):
        if line.startswith("#"):
            continue
        row = line.strip().split("\t")
        if row[2] != "gene":
            continue
        chrom, start, end, strand, attrs = (
            row[0],
            int(row[3]),
            int(row[4]),
            row[6],
            row[8],
        )
        tss = start if strand == "+" else end

        attrs_dict = get_gtf_attributes_dict(attrs)
        gene_id = attrs_dict["gene_id"]
        gene_name = attrs_dict.get("gene_name", None)
        if gene_name is None:
            gene_name = gene_id.split(".")[0]

        g = Gene(gene_id, gene_name.lower(), chrom, tss, strand)
        genes_by_id[gene_id] = g

    return genes_by_id


def get_egenes(cis_eqtl_path: str) -> set[str]:
    df = pd.read_csv(cis_eqtl_path, sep="\t", index_col=0)
    df = df[df["pvalue"] < 0.05].copy()
    return set(df["GENE_ID"].tolist())


def get_top_eqtls(cis_eqtl_path: str) -> dict[str, int]:
    """
    Returns:
        rsids: gene_id -> rsid of top eQTL
        distances: gene_id -> distance of top eQTL
    """
    df = pd.read_csv(cis_eqtl_path, sep="\t", index_col=0)
    df = df[df["pvalue"] < 0.05].copy()

    # For each gene in GENE_ID, sort SNPs by (log10pvalue: descending, distance: ascending) and get
    # the rsid and distance of the top SNP
    rsids = {}
    distances = {}

    for gene_id, df_gene in df.groupby("GENE_ID"):
        df_gene = df_gene.sort_values(
            ["log10pvalue", "distance"], ascending=[False, True]
        )
        rsids[gene_id] = df_gene.iloc[0].name
        if not rsids[gene_id].startswith("rs"):
            print(f"Warning: {gene_id} has top rsid {rsids[gene_id]}")
        distances[gene_id] = df_gene.iloc[0]["distance"]

    return (rsids, distances)


def get_our_gene_names() -> dict[str, str]:
    df = pd.read_csv(
        OUR_GENES_METADATA_PATH,
        names=["gene_id", "chrom", "tss", "gene_name", "strand"],
    )
    df["gene_name"] = df["gene_name"].fillna(df["gene_id"])
    df["gene_name"] = df["gene_name"].str.lower()
    return df.set_index("gene_id")["gene_name"].to_dict()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_csv_path", type=str)
    parser.add_argument("output_csv_path", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    genes_by_id = load_genes()
    eur_egenes = get_egenes(EUR_CIS_EQTL_PATH)
    yri_egenes = get_egenes(YRI_CIS_EQTL_PATH)
    top_eur_eqtl_rsids, top_eur_eqtl_distances = get_top_eqtls(EUR_CIS_EQTL_PATH)
    top_yri_eqtl_rsids, top_yri_eqtl_distances = get_top_eqtls(YRI_CIS_EQTL_PATH)
    gene_id_to_our_gene_names = get_our_gene_names()

    counts_df = pd.read_csv(
        args.input_csv_path, encoding="utf-8", index_col=0
    )  # [genes, samples]
    counts_df["stable_id"] = counts_df.index.str.split(".").str[0]
    counts_df["gencode_v12_gene_name"] = counts_df.index.map(
        lambda x: genes_by_id[x].gene_name
    )
    counts_df["our_gene_name"] = counts_df["stable_id"].map(
        gene_id_to_our_gene_names.get
    )

    counts_df["EUR_eGene"] = counts_df.index.isin(eur_egenes)
    counts_df["YRI_eGene"] = counts_df.index.isin(yri_egenes)
    counts_df["top_EUR_eqtl_rsid"] = counts_df.index.map(top_eur_eqtl_rsids.get)
    counts_df["top_YRI_eqtl_rsid"] = counts_df.index.map(top_yri_eqtl_rsids.get)
    counts_df["top_EUR_eqtl_distance"] = counts_df.index.map(top_eur_eqtl_distances.get)
    counts_df["top_YRI_eqtl_distance"] = counts_df.index.map(top_yri_eqtl_distances.get)

    print("Number of EUR eGenes:", counts_df["EUR_eGene"].sum())
    print("Number of YRI eGenes:", counts_df["YRI_eGene"].sum())

    # Assert that our_gene_name is not None for EUR eGenes
    assert counts_df[counts_df["EUR_eGene"]]["our_gene_name"].notnull().all()

    counts_df.to_csv(args.output_csv_path)


if __name__ == "__main__":
    main()
