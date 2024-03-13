from collections import defaultdict

import h5py
import pandas as pd


TRAIN_H5_PATH = "../finetuning/data/h5_bins_384_chrom_split/train.h5"
VAL_H5_PATH = "../finetuning/data/h5_bins_384_chrom_split/val.h5"
TEST_H5_PATH = "../finetuning/data/h5_bins_384_chrom_split/test.h5"
OUTPUT_PATH = "sample_splits.h5_bins_384_chrom_split.csv"


def get_samples(h5_split_path: str) -> dict[str, set]:
    """
    Returns a dictionary mapping genes to a set of samples for that gene found
    within the split.
    """
    gene_to_samples = defaultdict(set)
    with h5py.File(h5_split_path, "r") as f:
        genes = f["genes"][:].astype(str)
        samples = f["samples"][:].astype(str)
        for (gene, sample) in zip(genes, samples):
            gene_to_samples[gene].add(sample)
    return gene_to_samples
    


def main():
    gene_to_train_samples = get_samples(TRAIN_H5_PATH)
    gene_to_val_samples = get_samples(VAL_H5_PATH)
    gene_to_test_samples = get_samples(TEST_H5_PATH)
    
    # Get a sorted list of all genes
    genes = gene_to_train_samples.keys() | gene_to_val_samples.keys() | gene_to_test_samples.keys()
    genes = sorted(list(genes))
    assert len(genes) == 500, f"Found {len(genes)} genes"
    
    # Get a sorted list of all samples
    train_samples = set.union(*gene_to_train_samples.values())
    val_samples = set.union(*gene_to_val_samples.values())
    test_samples = set.union(*gene_to_test_samples.values())
    samples = sorted(list(train_samples | val_samples | test_samples))
    assert len(samples) == 421, f"Found {len(samples)} samples"
    
    # Collate sample splits into a matrix
    all_sample_splits = []
    for g in genes:
        gene_sample_splits = []
        for s in samples:
            if s in gene_to_train_samples[g]:
                gene_sample_splits.append("train")
            elif s in gene_to_val_samples[g]:
                gene_sample_splits.append("val")
            elif s in gene_to_test_samples[g]:
                gene_sample_splits.append("test")
            else:
                raise Exception(f"Sample {s} was not found in any split for gene {g}")
        all_sample_splits.append(gene_sample_splits)
    
    sample_splits_df = pd.DataFrame(all_sample_splits, index=genes, columns=samples)
    sample_splits_df.to_csv(OUTPUT_PATH)
    

if __name__ == "__main__":
    main()