import h5py
import os
from argparse import ArgumentParser
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

ENFORMER_SEQ_LEN = 393216
VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("context_length", type=int)
    parser.add_argument("--enformer_consensus_dir", type=str, default="/clusterfs/nilah/personalized_expression/consensus_seqs/enformer")
    parser.add_argument("--genes_csv_path", type=str, default="/clusterfs/nilah/personalized_expression/eur_eqtl_gene_list.csv")
    parser.add_argument("--expression_csv_path", type=str, default="/clusterfs/nilah/ruchir/src/personalized_genomes/connie_data/all_geuvadis_df.csv")
    return parser.parse_args()


def read_genes_file(genes_csv_path: str) -> pd.DataFrame:
    genes_df = pd.read_csv(genes_csv_path, names=["ensembl_id", "chrom", "TSS", "gene_symbol", "strand"], index_col=False)
    genes_df["gene_symbol"] = genes_df["gene_symbol"].fillna(genes_df["ensembl_id"])
    genes_df = genes_df.set_index("gene_symbol")
    genes_df.index = genes_df.index.str.lower()
    return genes_df


def get_seq_from_fasta(fasta_path: str, context_length: int, strand: str) -> tuple[np.ndarray, int, int]:
    record = SeqIO.read(fasta_path, "fasta")
    chrom, interval = record.name.split(":")
    seq = str(record.seq).upper()
    
    if interval.startswith("-"):
        # Beginning is truncated
        enformer_start = -int(pos.split("-")[0])
        enformer_end = int(pos.split("-")[1])
        assert enformer_end - enformer_start + 1 == ENFORMER_SEQ_LEN
        seq = "N" * (ENFORMER_SEQ_LEN - len(seq)) + seq
    else:
        enformer_start, enformer_end = map(int, interval.split("-"))
        assert enformer_end - enformer_start + 1 == ENFORMER_SEQ_LEN
        if len(seq) < ENFORMER_SEQ_LEN:
            seq += "N" * (ENFORMER_SEQ_LEN - len(seq))
    
    assert len(seq) == ENFORMER_SEQ_LEN
    assert context_length % 2 == 0
    
    # TSS will be centered with less sequence upstream than downstream
    if strand == "+":
        start = len(seq) // 2 - context_length // 2 + 1
    else:
        start = len(seq) // 2 - context_length
    end = start + context_length
    seq = seq[start: end]
    
    seq_arr = [VOCAB[bp] for bp in seq]
    return seq_arr, start, end


def run_on_gene(
    args,
    genes_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    gene: str
):
    strand = genes_df.loc[gene]["strand"]
    samples = expression_df.index
   
    seqs_arr = np.full((len(samples), 2, args.context_length), np.nan, dtype=np.uint8)
    expr_arr = np.full((len(samples)), np.nan, dtype=np.float32)
    start, end = None, None
    
    for sample_idx, sample in enumerate(samples):
        h1_fasta_path = os.path.join(args.enformer_consensus_dir, gene, "samples", f"{sample}.1pIu.fa")
        h2_fasta_path = os.path.join(args.enformer_consensus_dir, gene, "samples", f"{sample}.2pIu.fa")
        seqs_arr[sample_idx, 0], start, end = get_seq_from_fasta(h1_fasta_path, args.context_length, strand)
        seqs_arr[sample_idx, 1], start, end = get_seq_from_fasta(h2_fasta_path, args.context_length, strand)
        expr_arr[sample_idx] = expression_df[gene][sample]
        
    hf = h5py.file(os.path.join(args.output_dir, gene), "w")
    hf.create_dataset("seqs", data=seqs_arr)
    hf.create_dataset("exprs", data=expr_arr)
    hf.create_dataset("gene", data=gene)
    hf.create_dataset("chrom", data=genes_df.loc[gene]["chrom"])
    hf.create_dataset("TSS", data=genes_df.loc[gene]["TSS"])
    hf.create_dataset("strand", data=strand)
    hf.create_dataset("start", data=start)
    hf.create_dataset("end", data=end)
    hf.close()
    
    
def main():
    args = parse_args()
    
    genes_df = read_genes_file(args.genes_csv_path)
    expression_df = pd.read_csv(args.expression_csv_path, index_col=0) # [samples, genes]
    assert set(expression_df.columns) == set(genes_df.index)
    
    os.makedirs(args.output_dir, exist_ok=True)
    Parallel(n_jobs=-1, verbose=10)(
        delayed(run_on_gene)(args, genes_df, expression_df, gene)
        for gene in expression_df.columns
    )
    

if __name__ == '__main__':
    main()