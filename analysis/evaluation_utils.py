from collections import defaultdict

import h5py
import numpy as np
import pandas as pd


def load_finetuned_model_predictions(npz_path: str) -> pd.DataFrame:
    data = np.load(npz_path)
    preds = data["preds"]
    genes = data["genes"]
    samples = data["samples"]

    unique_genes = sorted(np.unique(genes))
    unique_samples = sorted(np.unique(samples))
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
    sample_to_idx = {s: i for i, s in enumerate(unique_samples)}

    mtx = np.full((len(genes), len(samples)), np.nan)
    for gene, sample, pred in zip(genes, samples, preds):
        mtx[gene_to_idx[gene], sample_to_idx[sample]] = pred
    return pd.DataFrame(mtx, index=unique_genes, columns=unique_samples)


def load_samples_per_gene(h5_path: str) -> dict[str, set]:
    samples_per_gene = defaultdict(set)
    with h5py.File(h5_path, "r") as f:
        genes = f["genes"][:].astype(str)
        samples = f["samples"][:].astype(str)
        for (g, s) in zip(genes, samples):
            samples_per_gene[g].add(s)
    return samples_per_gene


def get_gene_to_class_map(
    fpath: str = "/data/yosef3/users/ruchir/finetuning-enformer/finetuning/data/h5_bins_384_chrom_split/gene_class.csv",
):
    df = pd.read_csv(fpath, index_col=0)
    return df["class"].to_dict()
