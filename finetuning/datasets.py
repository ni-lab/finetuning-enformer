import glob
import json
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
import utils
from enformer_pytorch.data import seq_indices_to_one_hot, str_to_one_hot
from torchdata.datapipes.iter import FileLister, FileOpener
from tqdm import tqdm


def subsample_gene_to_idxs(
    gene_to_idxs: dict[str, np.ndarray],
    subsample_ratio: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Subsample the gene_to_idxs dictionary so that we only see a fraction of the data.
    """
    # First, shuffle the indices for each gene
    for g in sorted(gene_to_idxs):
        gene_to_idxs[g] = rng.permutation(gene_to_idxs[g])

    # Now, select the first subsample_ratio * len(gene_to_idxs[g]) indices for each gene\
    subsampled_gene_to_idxs = {}
    for g in gene_to_idxs:
        n_samples = int(len(gene_to_idxs[g]) * subsample_ratio)
        subsampled_gene_to_idxs[g] = gene_to_idxs[g][:n_samples]
    return subsampled_gene_to_idxs


class RefDataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str):
        data = np.load(filepath)
        self.genes = data["genes"]
        self.ref_seqs = data["ref_seqs"]
        assert self.genes.size == self.ref_seqs.shape[0]

        self.seq_idx_embedder = utils.create_seq_idx_embedder()

    def __len__(self):
        return self.ref_seqs.shape[0]

    def __getitem__(self, idx):
        seq = self.ref_seqs[idx]  # (L,)
        seq = np.repeat(seq[np.newaxis, :], 2, axis=0)  # (2, L)
        seq = self.seq_idx_embedder[seq]
        return {"seq": seq}


class SampleH5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: str,
        seqlen: int,
        prefetch_seqs: bool = False,
        reverse_complement_prob: float = 0.0,
        random_shift: bool = False,
        random_shift_max: int = 0,
        return_reverse_complement: bool = False,
        shift_max: int = 0,
        remove_rare_variants: bool = False,
        rare_variant_af_threshold: float = 0.05,
        train_h5_path_for_af_computation: str = None,
        force_recompute_afs: bool = False,
    ):
        """
        If prefetch_seqs is True, then all sequences are loaded into memory. This makes initialization
        very slow (~15 minutes for the training set of h5_bins_384), but speeds up __getitem__.
        We recommend setting this to True only if you have a lot of memory (> 200GB per process).
        """
        super().__init__()
        assert seqlen % 128 == 0

        self.h5_file = h5py.File(h5_path, "r")
        self.seqlen = seqlen

        # used in train mode
        self.reverse_complement_prob = reverse_complement_prob
        self.random_shift = random_shift
        self.random_shift_max = random_shift_max

        # used in val mode
        self.return_reverse_complement = return_reverse_complement
        self.shift_max = shift_max

        if self.return_reverse_complement or self.reverse_complement_prob > 0.0:
            assert (
                self.return_reverse_complement and self.reverse_complement_prob == 0.0
            ) or (
                not self.return_reverse_complement
                and self.reverse_complement_prob > 0.0
            ), "return_reverse_complement and reverse_complement_prob must be mutually exclusive - either return_reverse_complement is True and reverse_complement_prob is 0.0 or return_reverse_complement is False and reverse_complement_prob is > 0.0"

        if self.random_shift or self.shift_max > 0:
            assert (self.random_shift and shift_max == 0) or (
                not self.random_shift and shift_max > 0
            ), "random_shift and shift_max must be mutually exclusive - either random_shift is True and shift_max is 0 or random_shift is False and shift_max is > 0"

        # Load everything into memory
        self.genes = self.h5_file["genes"][:].astype(str)
        self.samples = self.h5_file["samples"][:].astype(str)
        if prefetch_seqs:
            self.seqs = self.h5_file["seqs"][:]  # (n_seqs, 2, length, 4)
        else:
            self.seqs = self.h5_file["seqs"]
        assert self.seqs.shape[2] >= self.seqlen
        self.Y = self.h5_file["Y"][:]
        self.Z = self.h5_file["Z"][:]
        self.percentiles = self.h5_file["P"][:]

        assert (
            self.genes.size
            == self.samples.size
            == self.seqs.shape[0]
            == self.Y.size
            == self.Z.size
            == self.percentiles.size
        )

        # compute allele freqs from the training set if we are removing rare variants
        if remove_rare_variants:
            assert train_h5_path_for_af_computation is not None
            self.afs = self.__compute_afs(
                train_h5_path_for_af_computation, force_recompute_afs
            )
            self.__filter_rare_variants(rare_variant_af_threshold)

    def __compute_afs(self, train_h5_path: str, force_recompute_afs: bool):
        """
        Compute allele freqs from the training set for each variant.
        Args:
            train_h5_path: path to the training h5 file
            force_recompute_afs: if True, recompute the allele freqs even if they are already exist
        Returns:
            allele_freqs: a dictionary where gene names are keys and values are numpy arrays of allele freqs
        """
        afs_cache_path = train_h5_path + ".afs.pkl"
        train_h5_file = h5py.File(train_h5_path, "r")
        train_genes = train_h5_file["genes"][:].astype(str)

        if os.path.exists(afs_cache_path) and not force_recompute_afs:
            with open(afs_cache_path, "rb") as f:
                afs = pickle.load(f)

            # check that the MAFs are computed for all genes
            assert set(afs.keys()) == set(np.unique(train_genes))

            print(f"Loaded cached allele freqs from {afs_cache_path}")

            return afs

        train_samples = train_h5_file["samples"][:].astype(str)
        train_seqs = train_h5_file["seqs"]
        assert train_seqs.shape[2] >= self.seqlen
        train_Y = train_h5_file["Y"][:]
        train_Z = train_h5_file["Z"][:]
        train_percentiles = train_h5_file["P"][:]

        assert (
            train_genes.size
            == train_samples.size
            == train_seqs.shape[0]
            == train_Y.size
            == train_Z.size
            == train_percentiles.size
        )

        print(f"Computing allele freqs from {train_h5_path}...")

        afs = {}
        for gene in tqdm(np.unique(train_genes)):
            gene_idxs = train_genes == gene
            gene_seqs = train_seqs[gene_idxs]  # (n_seqs, 2, length, 4)
            gene_allele_seqs = gene_seqs.reshape(
                -1, self.seqlen, 4
            )  # (n_seqs * 2, length, 4)
            gene_allele_counts = gene_allele_seqs.sum(axis=0)  # (length, 4)
            gene_allele_freqs = (
                gene_allele_counts / gene_allele_seqs.shape[0]
            )  # (length, 4)
            afs[gene] = gene_allele_freqs

        with open(afs_cache_path, "wb+") as f:
            pickle.dump(afs, f)

        print(f"Computed allele freqs and saved to {afs_cache_path}")

        return afs

    def __filter_rare_variants(self, af_threshold: float):
        """
        Filter out rare variants from the dataset by replacing the rare variants with the major allele.
        Args:
            af_threshold: allele frequency threshold below which variants are considered rare
        """
        print(f"Filtering out rare variants with AF < {af_threshold}...")
        for gene in self.afs:
            gene_afs = self.afs[gene]
            rare_variant_mask = gene_afs < af_threshold
            gene_seqs = self.seqs[self.genes == gene]  # (n_seqs, 2, length, 4)
            gene_seqs = gene_seqs.reshape(-1, self.seqlen, 4)  # (n_seqs * 2, length, 4)
            gene_seqs[:, rare_variant_mask] = 0  # zero out the rare variants
            # at positions where we zeroed out the rare variants, we need to update them to the major allele
            major_allele = gene_afs.argmax(axis=-1)
            positions_that_were_rare = (
                gene_seqs.sum(axis=-1) == 0
            )  # (n_seqs * 2, length)
            gene_seqs[positions_that_were_rare, major_allele] = 1
            self.seqs[self.genes == gene] = gene_seqs.reshape(-1, 2, self.seqlen, 4)
        print("Done filtering rare variants.")

    def get_total_n_bins(self):
        return self.seqlen // 128

    def __len__(self):
        return (
            self.seqs.shape[0]
            * (2 if self.return_reverse_complement else 1)
            * ((2 * self.shift_max) + 1)
        )

    def __shorten_seq(self, seq):
        """
        seq: (2, seqlen, 4)
        """
        if seq.shape[1] == self.seqlen:
            return seq
        start_idx = (seq.shape[1] - self.seqlen) // 2
        end_idx = start_idx + self.seqlen
        return seq[:, start_idx:end_idx, :]

    def __getitem__(self, idx):
        if self.return_reverse_complement:
            true_idx = idx // (2 * ((2 * self.shift_max) + 1))
        else:
            true_idx = idx // ((2 * self.shift_max) + 1)

        seq = self.__shorten_seq(self.seqs[true_idx]).astype(np.float32)
        y = self.Y[true_idx]
        z = self.Z[true_idx]

        reverse_complement = False
        if self.reverse_complement_prob > 0.0:
            if np.random.random() < self.reverse_complement_prob:
                seq = np.flip(seq, axis=(-1, -2)).copy()
                reverse_complement = True
        elif self.return_reverse_complement and (idx % 2 == 1):
            seq = np.flip(seq, axis=(-1, -2)).copy()
            reverse_complement = True

        if self.random_shift:
            shift = np.random.randint(-self.random_shift_max, self.random_shift_max + 1)
            seq = np.roll(seq, shift, axis=-2)
            # zero out the shifted positions
            if shift > 0:
                seq[:, :shift, :] = 0
            elif shift < 0:
                seq[:, shift:, :] = 0
        elif self.shift_max > 0:
            shift = ((idx // 2) % (2 * self.shift_max + 1)) - self.shift_max
            seq = np.roll(seq, shift, axis=-2)
            if shift > 0:
                seq[:, :shift, :] = 0
            elif shift < 0:
                seq[:, shift:, :] = 0
        else:
            shift = 0

        return {
            "seq": seq,
            "y": y,
            "z": z,
            "gene": self.genes[true_idx],
            "sample": self.samples[true_idx],
            "true_idx": true_idx,
            "reverse_complement": reverse_complement,
            "shift": shift,
        }


class PairwiseClassificationH5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: str,
        n_pairs_per_gene: int,
        seqlen: int,
        min_percentile_diff: float = 25.0,
        prefetch_seqs: bool = False,
        random_seed: int = 42,
        return_reverse_complement: bool = False,
        shift_max: int = 0,
    ):
        """
        If prefetch_seqs is True, then all sequences are loaded into memory. This makes initialization
        very slow (~15 minutes for the training set of h5_bins_384), but speeds up __getitem__.
        We recommend setting this to True only if you have a lot of memory (> 200GB per process).
        """
        super().__init__()
        assert seqlen % 128 == 0

        self.h5_file = h5py.File(h5_path, "r")
        self.n_pairs_per_gene = n_pairs_per_gene
        self.seqlen = seqlen
        self.min_percentile_diff = min_percentile_diff
        self.rng = np.random.default_rng(random_seed)
        self.return_reverse_complement = return_reverse_complement
        self.shift_max = shift_max

        # Load everything into memory
        self.genes = self.h5_file["genes"][:].astype(str)
        self.samples = self.h5_file["samples"][:].astype(str)
        if prefetch_seqs:
            self.seqs = self.h5_file["seqs"][:]  # (n_seqs, 2, length, 4)
        else:
            self.seqs = self.h5_file["seqs"]
        assert self.seqs.shape[2] >= self.seqlen
        self.Y = self.h5_file["Y"][:]
        self.Z = self.h5_file["Z"][:]
        self.percentiles = self.h5_file["P"][:]

        assert (
            self.genes.size
            == self.samples.size
            == self.seqs.shape[0]
            == self.Y.size
            == self.Z.size
            == self.percentiles.size
        )

        self.pairs = self.__sample_pairs()

    def __sample_pairs(self):
        unique_genes = sorted(np.unique(self.genes))
        gene_to_idxs = {g: np.where(self.genes == g)[0] for g in unique_genes}

        pairs = []
        for g in unique_genes:
            g_idxs = gene_to_idxs[g]
            g_pairs = []
            while len(g_pairs) < self.n_pairs_per_gene:
                idxs = self.rng.choice(g_idxs, size=2, replace=False)
                percentile_diff = np.abs(
                    self.percentiles[idxs[0]] - self.percentiles[idxs[1]]
                )
                if percentile_diff > self.min_percentile_diff:
                    g_pairs.append(idxs)
            pairs.extend(g_pairs)
        pairs = np.array(pairs)
        self.rng.shuffle(pairs)
        return pairs

    def get_total_n_bins(self):
        return self.seqlen // 128

    def __len__(self):
        return (
            self.pairs.shape[0]
            * (2 if self.return_reverse_complement else 1)
            * ((2 * self.shift_max) + 1)
        )

    def __shorten_seq(self, seq):
        """
        seq: (2, seqlen, 4)
        """
        if seq.shape[1] == self.seqlen:
            return seq
        start_idx = (seq.shape[1] - self.seqlen) // 2
        end_idx = start_idx + self.seqlen
        return seq[:, start_idx:end_idx, :]

    def __getitem__(self, idx):
        if self.return_reverse_complement:
            true_idx = idx // (2 * ((2 * self.shift_max) + 1))
        else:
            true_idx = idx // ((2 * self.shift_max) + 1)

        seq_idx1, seq_idx2 = self.pairs[true_idx]
        assert self.genes[seq_idx1] == self.genes[seq_idx2]
        assert self.samples[seq_idx1] != self.samples[seq_idx2]

        seq1 = self.__shorten_seq(self.seqs[seq_idx1].astype(np.float32))
        seq2 = self.__shorten_seq(self.seqs[seq_idx2].astype(np.float32))

        if self.return_reverse_complement and idx % 2 == 1:
            seq1 = np.flip(seq1, axis=(-1, -2)).copy()
            seq2 = np.flip(seq2, axis=(-1, -2)).copy()
            reverse_complement = True
        else:
            reverse_complement = False

        if self.shift_max > 0:
            shift = ((idx // 2) % (2 * self.shift_max + 1)) - self.shift_max
            seq1 = np.roll(seq1, shift, axis=-2)
            seq2 = np.roll(seq2, shift, axis=-2)
            if shift > 0:
                seq1[:, :shift, :] = 0
                seq2[:, :shift, :] = 0
            elif shift < 0:
                seq1[:, shift:, :] = 0
                seq2[:, shift:, :] = 0
        else:
            shift = 0

        Y = int(self.Y[seq_idx1] >= self.Y[seq_idx2])
        return {
            "seq1": seq1,
            "seq2": seq2,
            "Y": Y,
            "gene": self.genes[seq_idx1],
            "sample1": self.samples[seq_idx1],
            "sample2": self.samples[seq_idx2],
            "true_idx": true_idx,
            "reverse_complement": reverse_complement,
            "shift": shift,
        }


class PairwiseClassificationH5DatasetDynamicSampling(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: str,
        n_pairs_per_gene: int,
        seqlen: int,
        min_percentile_diff: float = 25.0,
        prefetch_seqs: bool = False,
        random_seed: int = 42,
        reverse_complement_prob: float = 0.0,
        random_shift: bool = False,
        random_shift_max: int = 0,
        subsample_ratio: float = 1.0,
    ):
        """
        If prefetch_seqs is True, then all sequences are loaded into memory. This makes initialization
        very slow (~15 minutes for the training set of h5_bins_384), but speeds up __getitem__.
        We recommend setting this to True only if you have a lot of memory (> 200GB per process).
        """
        super().__init__()
        assert seqlen % 128 == 0

        self.h5_file = h5py.File(h5_path, "r")
        self.n_pairs_per_gene = n_pairs_per_gene
        self.seqlen = seqlen
        self.min_percentile_diff = min_percentile_diff
        self.rng = np.random.default_rng(random_seed)
        self.reverse_complement_prob = reverse_complement_prob
        self.random_shift = random_shift
        self.random_shift_max = random_shift_max

        # Load everything into memory
        self.genes = self.h5_file["genes"][:].astype(str)
        self.samples = self.h5_file["samples"][:].astype(str)
        if prefetch_seqs:
            self.seqs = self.h5_file["seqs"][:]  # (n_seqs, 2, length, 4)
        else:
            self.seqs = self.h5_file["seqs"]
        assert self.seqs.shape[2] >= self.seqlen
        self.Y = self.h5_file["Y"][:]
        self.Z = self.h5_file["Z"][:]
        self.percentiles = self.h5_file["P"][:]

        assert (
            self.genes.size
            == self.samples.size
            == self.seqs.shape[0]
            == self.Y.size
            == self.Z.size
            == self.percentiles.size
        )

        self.unique_genes = sorted(np.unique(self.genes))
        self.gene_to_idxs = {g: np.where(self.genes == g)[0] for g in self.unique_genes}
        if subsample_ratio < 1.0:
            self.gene_to_idxs = subsample_gene_to_idxs(
                self.gene_to_idxs, subsample_ratio, self.rng
            )

    def get_total_n_bins(self):
        return self.seqlen // 128

    def __len__(self):
        return self.n_pairs_per_gene * len(np.unique(self.genes))

    def __shorten_seq(self, seq):
        """
        seq: (2, seqlen, 4)
        """
        if seq.shape[1] == self.seqlen:
            return seq
        start_idx = (seq.shape[1] - self.seqlen) // 2
        end_idx = start_idx + self.seqlen
        return seq[:, start_idx:end_idx, :]

    def __getitem__(self, idx):
        chosen_gene = idx % len(self.unique_genes)
        chosen_gene = self.unique_genes[chosen_gene]
        gene_idxs = self.gene_to_idxs[chosen_gene]
        percentile_diff = -1
        idx1, idx2 = None, None
        while percentile_diff <= self.min_percentile_diff:
            idx1, idx2 = self.rng.choice(gene_idxs, size=2, replace=False)
            percentile_diff = np.abs(self.percentiles[idx1] - self.percentiles[idx2])

        assert self.genes[idx1] == self.genes[idx2]
        assert self.samples[idx1] != self.samples[idx2]

        seq1 = self.__shorten_seq(self.seqs[idx1].astype(np.float32))
        seq2 = self.__shorten_seq(self.seqs[idx2].astype(np.float32))

        if self.rng.random() < self.reverse_complement_prob:
            # flip along the sequence length and the one-hot encoding axis -
            # this is the reverse complement as the one-hot encoding is in the order ACGT
            seq1 = np.flip(seq1, axis=(-1, -2)).copy()
            seq2 = np.flip(seq2, axis=(-1, -2)).copy()

        if self.random_shift:
            shift = self.rng.integers(-self.random_shift_max, self.random_shift_max + 1)
            seq1 = np.roll(seq1, shift, axis=-2)
            seq2 = np.roll(seq2, shift, axis=-2)

            # zero out the shifted positions
            if shift > 0:
                seq1[:, :shift, :] = 0
                seq2[:, :shift, :] = 0
            elif shift < 0:
                seq1[:, shift:, :] = 0
                seq2[:, shift:, :] = 0
        else:
            shift = 0

        Y = int(self.Y[idx1] >= self.Y[idx2])
        return {"seq1": seq1, "seq2": seq2, "Y": Y}


class PairwiseRegressionOnCountsH5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: str,
        n_pairs_per_gene: int,
        seqlen: int,
        prefetch_seqs: bool = False,
        random_seed: int = 42,
        return_reverse_complement: bool = False,
        shift_max: int = 0,
    ):
        """
        If prefetch_seqs is True, then all sequences are loaded into memory. This makes initialization
        very slow (~15 minutes for the training set of h5_bins_384), but speeds up __getitem__.
        We recommend setting this to True only if you have a lot of memory (> 200GB per process).
        """
        super().__init__()
        assert seqlen % 128 == 0

        self.h5_file = h5py.File(h5_path, "r")
        self.n_pairs_per_gene = n_pairs_per_gene
        self.seqlen = seqlen
        self.rng = np.random.default_rng(random_seed)
        self.return_reverse_complement = return_reverse_complement
        self.shift_max = shift_max

        # Load everything into memory
        self.genes = self.h5_file["genes"][:].astype(str)
        self.samples = self.h5_file["samples"][:].astype(str)
        if prefetch_seqs:
            self.seqs = self.h5_file["seqs"][:]
        else:
            self.seqs = self.h5_file["seqs"]
        assert self.seqs.shape[2] >= self.seqlen
        self.Y = self.h5_file["Y"][:]

        assert self.genes.size == self.samples.size == self.seqs.shape[0] == self.Y.size

        self.pairs = self.__sample_pairs()

    def get_total_n_bins(self):
        return self.seqlen // 128

    def __len__(self):
        return (
            self.n_pairs_per_gene
            * len(np.unique(self.genes))
            * (2 if self.return_reverse_complement else 1)
            * ((2 * self.shift_max) + 1)
        )

    def __shorten_seq(self, seq):
        """
        seq: (2, seqlen, 4)
        """
        if seq.shape[1] == self.seqlen:
            return seq
        start_idx = (seq.shape[1] - self.seqlen) // 2
        end_idx = start_idx + self.seqlen
        return seq[:, start_idx:end_idx, :]

    def __sample_pairs(self):
        unique_genes = sorted(np.unique(self.genes))
        gene_to_idxs = {g: np.where(self.genes == g)[0] for g in unique_genes}

        pairs = []
        for g in unique_genes:
            g_idxs = gene_to_idxs[g]
            pairs.extend(
                [
                    self.rng.choice(g_idxs, size=2, replace=False)
                    for _ in range(self.n_pairs_per_gene)
                ]
            )
        pairs = np.array(pairs)
        self.rng.shuffle(pairs)
        return pairs

    def __getitem__(self, idx):
        if self.return_reverse_complement:
            true_idx = idx // (2 * ((2 * self.shift_max) + 1))
        else:
            true_idx = idx // ((2 * self.shift_max) + 1)

        idx1, idx2 = self.pairs[true_idx]
        assert self.genes[idx1] == self.genes[idx2]
        assert self.samples[idx1] != self.samples[idx2]

        seq1 = self.__shorten_seq(self.seqs[idx1].astype(np.float32))
        seq2 = self.__shorten_seq(self.seqs[idx2].astype(np.float32))

        if self.return_reverse_complement and idx % 2 == 1:
            seq1 = np.flip(seq1, axis=(-1, -2)).copy()
            seq2 = np.flip(seq2, axis=(-1, -2)).copy()
            reverse_complement = True
        else:
            reverse_complement = False

        if self.shift_max > 0:
            shift = ((idx // 2) % (2 * self.shift_max + 1)) - self.shift_max
            seq1 = np.roll(seq1, shift, axis=-2)
            seq2 = np.roll(seq2, shift, axis=-2)
            if shift > 0:
                seq1[:, :shift, :] = 0
                seq2[:, :shift, :] = 0
            elif shift < 0:
                seq1[:, shift:, :] = 0
                seq2[:, shift:, :] = 0
        else:
            shift = 0

        return {
            "seq1": seq1,
            "seq2": seq2,
            "Y1": self.Y[idx1],
            "Y2": self.Y[idx2],
            "gene": self.genes[idx1],
            "sample1": self.samples[idx1],
            "sample2": self.samples[idx2],
            "true_idx": true_idx,
            "reverse_complement": reverse_complement,
            "shift": shift,
        }


class PairwiseRegressionOnCountsH5DatasetDynamicSampling(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: str,
        n_pairs_per_gene: int,
        seqlen: int,
        prefetch_seqs: bool = False,
        random_seed: int = 42,
        reverse_complement_prob: float = 0.0,
        random_shift: bool = False,
        random_shift_max: int = 0,
        subsample_ratio: float = 1.0,
    ):
        """
        If prefetch_seqs is True, then all sequences are loaded into memory. This makes initialization
        very slow (~15 minutes for the training set of h5_bins_384), but speeds up __getitem__.
        We recommend setting this to True only if you have a lot of memory (> 200GB per process).
        """
        super().__init__()
        assert seqlen % 128 == 0

        self.h5_file = h5py.File(h5_path, "r")
        self.n_pairs_per_gene = n_pairs_per_gene
        self.seqlen = seqlen
        self.rng = np.random.default_rng(random_seed)
        self.reverse_complement_prob = reverse_complement_prob
        self.random_shift = random_shift
        self.random_shift_max = random_shift_max

        # Load everything into memory
        self.genes = self.h5_file["genes"][:].astype(str)
        self.samples = self.h5_file["samples"][:].astype(str)
        if prefetch_seqs:
            self.seqs = self.h5_file["seqs"][:]  # (n_seqs, 2, length, 4)
        else:
            self.seqs = self.h5_file["seqs"]
        assert self.seqs.shape[2] >= self.seqlen
        self.Y = self.h5_file["Y"][:]

        assert self.genes.size == self.samples.size == self.seqs.shape[0] == self.Y.size

        self.unique_genes = sorted(np.unique(self.genes))
        self.gene_to_idxs = {g: np.where(self.genes == g)[0] for g in self.unique_genes}
        if subsample_ratio < 1.0:
            self.gene_to_idxs = subsample_gene_to_idxs(
                self.gene_to_idxs, subsample_ratio, self.rng
            )

    def get_total_n_bins(self):
        return self.seqlen // 128

    def __len__(self):
        return self.n_pairs_per_gene * len(np.unique(self.genes))

    def __shorten_seq(self, seq):
        """
        seq: (2, seqlen, 4)
        """
        if seq.shape[1] == self.seqlen:
            return seq
        start_idx = (seq.shape[1] - self.seqlen) // 2
        end_idx = start_idx + self.seqlen
        return seq[:, start_idx:end_idx, :]

    def __getitem__(self, idx):
        chosen_gene = self.unique_genes[idx % len(self.unique_genes)]
        idx1, idx2 = self.rng.choice(
            self.gene_to_idxs[chosen_gene], size=2, replace=False
        )
        assert self.genes[idx1] == self.genes[idx2]
        assert self.samples[idx1] != self.samples[idx2]

        seq1 = self.__shorten_seq(self.seqs[idx1].astype(np.float32))
        seq2 = self.__shorten_seq(self.seqs[idx2].astype(np.float32))

        if self.rng.random() < self.reverse_complement_prob:
            # flip along the sequence length and the one-hot encoding axis -
            # this is the reverse complement as the one-hot encoding is in the order ACGT
            seq1 = np.flip(seq1, axis=(-1, -2)).copy()
            seq2 = np.flip(seq2, axis=(-1, -2)).copy()

        if self.random_shift:
            shift = self.rng.integers(-self.random_shift_max, self.random_shift_max + 1)
            seq1 = np.roll(seq1, shift, axis=-2)
            seq2 = np.roll(seq2, shift, axis=-2)

            # zero out the shifted positions
            if shift > 0:
                seq1[:, :shift, :] = 0
                seq2[:, :shift, :] = 0
            elif shift < 0:
                seq1[:, shift:, :] = 0
                seq2[:, shift:, :] = 0
        else:
            shift = 0

        return {"seq1": seq1, "seq2": seq2, "Y1": self.Y[idx1], "Y2": self.Y[idx2]}


class PairwiseRegressionH5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: str,
        n_pairs_per_gene: int,
        seqlen: int,
        prefetch_seqs: bool = False,
        random_seed: int = 42,
        return_reverse_complement: bool = False,
        shift_max: int = 0,
    ):
        """
        If prefetch_seqs is True, then all sequences are loaded into memory. This makes initialization
        very slow (~15 minutes for the training set of h5_bins_384), but speeds up __getitem__.
        We recommend setting this to True only if you have a lot of memory (> 200GB per process).
        """
        super().__init__()
        assert seqlen % 128 == 0

        self.h5_file = h5py.File(h5_path, "r")
        self.n_pairs_per_gene = n_pairs_per_gene
        self.seqlen = seqlen
        self.rng = np.random.default_rng(random_seed)
        self.return_reverse_complement = return_reverse_complement
        self.shift_max = shift_max

        # Load everything into memory
        self.genes = self.h5_file["genes"][:].astype(str)
        self.samples = self.h5_file["samples"][:].astype(str)
        if prefetch_seqs:
            self.seqs = self.h5_file["seqs"][:]  # (n_seqs, 2, length, 4)
        else:
            self.seqs = self.h5_file["seqs"]
        assert self.seqs.shape[2] >= self.seqlen
        self.Y = self.h5_file["Y"][:]
        self.Z = self.h5_file["Z"][:]
        self.percentiles = self.h5_file["P"][:]

        assert (
            self.genes.size
            == self.samples.size
            == self.seqs.shape[0]
            == self.Y.size
            == self.Z.size
            == self.percentiles.size
        )

        self.pairs = self.__sample_pairs()

    def __sample_pairs(self):
        unique_genes = sorted(np.unique(self.genes))
        gene_to_idxs = {g: np.where(self.genes == g)[0] for g in unique_genes}

        pairs = []
        for g in unique_genes:
            g_idxs = gene_to_idxs[g]
            pairs.extend(
                [
                    self.rng.choice(g_idxs, size=2, replace=False)
                    for _ in range(self.n_pairs_per_gene)
                ]
            )
        pairs = np.array(pairs)
        self.rng.shuffle(pairs)
        return pairs

    def get_total_n_bins(self):
        return self.seqlen // 128

    def __len__(self):
        return (
            self.pairs.shape[0]
            * (2 if self.return_reverse_complement else 1)
            * ((2 * self.shift_max) + 1)
        )

    def __shorten_seq(self, seq):
        """
        seq: (2, seqlen, 4)
        """
        if seq.shape[1] == self.seqlen:
            return seq
        start_idx = (seq.shape[1] - self.seqlen) // 2
        end_idx = start_idx + self.seqlen
        return seq[:, start_idx:end_idx, :]

    def __getitem__(self, idx):
        if self.return_reverse_complement:
            true_idx = idx // (2 * ((2 * self.shift_max) + 1))
        else:
            true_idx = idx // ((2 * self.shift_max) + 1)

        seq_idx1, seq_idx2 = self.pairs[true_idx]
        assert self.genes[seq_idx1] == self.genes[seq_idx2]
        assert self.samples[seq_idx1] != self.samples[seq_idx2]

        seq1 = self.__shorten_seq(self.seqs[seq_idx1].astype(np.float32))
        seq2 = self.__shorten_seq(self.seqs[seq_idx2].astype(np.float32))

        if self.return_reverse_complement and idx % 2 == 1:
            seq1 = np.flip(seq1, axis=(-1, -2)).copy()
            seq2 = np.flip(seq2, axis=(-1, -2)).copy()
            reverse_complement = True
        else:
            reverse_complement = False

        if self.shift_max > 0:
            shift = ((idx // 2) % (2 * self.shift_max + 1)) - self.shift_max
            seq1 = np.roll(seq1, shift, axis=-2)
            seq2 = np.roll(seq2, shift, axis=-2)
            if shift > 0:
                seq1[:, :shift, :] = 0
                seq2[:, :shift, :] = 0
            elif shift < 0:
                seq1[:, shift:, :] = 0
                seq2[:, shift:, :] = 0
        else:
            shift = 0

        z_diff = self.Z[seq_idx1] - self.Z[seq_idx2]
        return {
            "seq1": seq1,
            "seq2": seq2,
            "z_diff": z_diff,
            "gene": self.genes[seq_idx1],
            "sample1": self.samples[seq_idx1],
            "sample2": self.samples[seq_idx2],
            "true_idx": true_idx,
            "reverse_complement": reverse_complement,
            "shift": shift,
        }


class PairwiseRegressionH5DatasetDynamicSampling(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: str,
        n_pairs_per_gene: int,
        seqlen: int,
        prefetch_seqs: bool = False,
        random_seed: int = 42,
        reverse_complement_prob: float = 0.0,
        random_shift: bool = False,
        random_shift_max: int = 0,
        subsample_ratio: float = 1.0,
    ):
        """
        If prefetch_seqs is True, then all sequences are loaded into memory. This makes initialization
        very slow (~15 minutes for the training set of h5_bins_384), but speeds up __getitem__.
        We recommend setting this to True only if you have a lot of memory (> 200GB per process).
        """
        super().__init__()
        assert seqlen % 128 == 0

        self.h5_file = h5py.File(h5_path, "r")
        self.n_pairs_per_gene = n_pairs_per_gene
        self.seqlen = seqlen
        self.rng = np.random.default_rng(random_seed)
        self.reverse_complement_prob = reverse_complement_prob
        self.random_shift = random_shift
        self.random_shift_max = random_shift_max

        # Load everything into memory
        self.genes = self.h5_file["genes"][:].astype(str)
        self.samples = self.h5_file["samples"][:].astype(str)
        if prefetch_seqs:
            self.seqs = self.h5_file["seqs"][:]  # (n_seqs, 2, length, 4)
        else:
            self.seqs = self.h5_file["seqs"]
        assert self.seqs.shape[2] >= self.seqlen
        self.Y = self.h5_file["Y"][:]
        self.Z = self.h5_file["Z"][:]
        self.percentiles = self.h5_file["P"][:]

        assert (
            self.genes.size
            == self.samples.size
            == self.seqs.shape[0]
            == self.Y.size
            == self.Z.size
            == self.percentiles.size
        )

        self.unique_genes = sorted(np.unique(self.genes))
        self.gene_to_idxs = {g: np.where(self.genes == g)[0] for g in self.unique_genes}
        if subsample_ratio < 1.0:
            self.gene_to_idxs = subsample_gene_to_idxs(
                self.gene_to_idxs, subsample_ratio, self.rng
            )

    def get_total_n_bins(self):
        return self.seqlen // 128

    def __len__(self):
        return self.n_pairs_per_gene * len(np.unique(self.genes))

    def __shorten_seq(self, seq):
        """
        seq: (2, seqlen, 4)
        """
        if seq.shape[1] == self.seqlen:
            return seq
        start_idx = (seq.shape[1] - self.seqlen) // 2
        end_idx = start_idx + self.seqlen
        return seq[:, start_idx:end_idx, :]

    def __getitem__(self, idx):
        chosen_gene = idx % len(self.unique_genes)
        chosen_gene = self.unique_genes[chosen_gene]
        gene_idxs = self.gene_to_idxs[chosen_gene]
        idx1, idx2 = self.rng.choice(gene_idxs, size=2, replace=False)
        assert self.genes[idx1] == self.genes[idx2]
        assert self.samples[idx1] != self.samples[idx2]

        seq1 = self.__shorten_seq(self.seqs[idx1].astype(np.float32))
        seq2 = self.__shorten_seq(self.seqs[idx2].astype(np.float32))

        if self.rng.random() < self.reverse_complement_prob:
            # flip along the sequence length and the one-hot encoding axis -
            # this is the reverse complement as the one-hot encoding is in the order ACGT
            seq1 = np.flip(seq1, axis=(-1, -2)).copy()
            seq2 = np.flip(seq2, axis=(-1, -2)).copy()

        if self.random_shift:
            shift = self.rng.integers(-self.random_shift_max, self.random_shift_max + 1)
            seq1 = np.roll(seq1, shift, axis=-2)
            seq2 = np.roll(seq2, shift, axis=-2)

            # zero out the shifted positions
            if shift > 0:
                seq1[:, :shift, :] = 0
                seq2[:, :shift, :] = 0
            elif shift < 0:
                seq1[:, shift:, :] = 0
                seq2[:, shift:, :] = 0
        else:
            shift = 0

        z_diff = self.Z[idx1] - self.Z[idx2]
        return {"seq1": seq1, "seq2": seq2, "z_diff": z_diff}


class EnformerDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_dir: str,
        species: str,
        split: str,
        reverse_complement: bool = False,
        random_shift: bool = False,
        half_precision: bool = False,
    ):
        super().__init__()

        assert split in [
            "train",
            "val",
            "test",
        ], "split must be one of train, val, test"

        assert species in [
            "human",
            "mouse",
        ], "species must be one of human, mouse"

        self.data_dir = data_dir
        self.species = species
        self.split = split
        self.reverse_complement = reverse_complement
        self.random_shift = random_shift
        self.half_precision = half_precision

        if self.split == "val" or self.split == "test":
            if self.reverse_complement:
                raise ValueError(
                    "reverse_complement must be False for val and test splits"
                )
            if self.random_shift:
                raise ValueError("random_shift must be False for val and test splits")
        else:
            if not (self.reverse_complement or self.random_shift):
                print(
                    "WARNING: reverse_complement and random_shift are both False for train split. Setting these to True can improve model performance."
                )

        self.enformer_species_data_dir = os.path.join(data_dir, species)

        # read data parameters
        data_stats_file = os.path.join(
            self.enformer_species_data_dir, "statistics.json"
        )
        with open(data_stats_file) as data_stats_open:
            self.data_stats = json.load(data_stats_open)

        self.seq_length = self.data_stats["seq_length"]
        self.seq_depth = self.data_stats.get("seq_depth", 4)
        self.seq_1hot = self.data_stats.get("seq_1hot", False)
        self.target_length = self.data_stats["target_length"]
        self.num_targets = self.data_stats["num_targets"]

        if split == "train":
            self.num_seqs = self.data_stats["train_seq"]
        elif split == "val":
            self.num_seqs = self.data_stats["valid_seqs"]
        elif split == "test":
            self.num_seqs = self.data_stats["test_seqs"]

        print("Main process started")

    def __iter__(self):
        all_files = glob.glob(
            os.path.join(
                self.enformer_species_data_dir, "tfrecords", f"{self.split}*.tfr"
            )
        )

        self.datapipe1 = FileLister(all_files)
        self.datapipe2 = FileOpener(self.datapipe1, mode="b")
        self.tfrecord_loader_dp = self.datapipe2.load_from_tfrecord()

        # get process rank and shard data according to rank
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            print(f"Worker {self.rank} started")
        else:
            self.world_size = 1
            self.rank = 0

        # to make sure that each worker gets the same number of examples, trim off the last few examples
        # that are not divisible by world_size
        num_iterable_examples = (self.num_seqs // self.world_size) * self.world_size

        counter = 0
        for example in self.tfrecord_loader_dp:
            counter += 1
            if counter > num_iterable_examples:
                break
            if counter % self.world_size != self.rank:
                continue

            sequence = np.frombuffer(example["sequence"][0], dtype="uint8").reshape(
                self.seq_length, self.seq_depth
            )
            targets = np.frombuffer(example["target"][0], dtype="float16").reshape(
                self.target_length, self.num_targets
            )

            if self.reverse_complement:
                coin_flip = np.random.choice([True, False])
                if coin_flip:
                    sequence = np.flip(sequence, axis=0)
                    targets = np.flip(targets, axis=0)

                    # order of bases is ACGT, so reverse-complement is just flipping the sequence
                    sequence = np.flip(sequence, axis=1)

            if self.random_shift:
                shift = np.random.randint(-3, 4)
                sequence = np.roll(sequence, shift, axis=0)
                # targets are not shifted
                # zero out the shifted positions
                if shift > 0:
                    sequence[:shift] = 0
                elif shift < 0:
                    sequence[shift:] = 0

            if self.half_precision:
                seq = torch.tensor(sequence.copy()).half()
                y = torch.tensor(targets.copy()).half()
            else:
                seq = torch.tensor(sequence.copy()).float()
                y = torch.tensor(targets.copy()).float()

            yield {"seq": seq, "y": y}
