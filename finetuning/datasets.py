import glob
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
import utils
from enformer_pytorch.data import seq_indices_to_one_hot, str_to_one_hot
from torchdata.datapipes.iter import FileLister, FileOpener


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str):
        data = np.load(filepath)
        self.genes = data["genes"]
        self.seqs = data["seqs"]
        self.samples = data["samples"]
        self.Y = data["Y"]
        self.Z = data["Z"]
        assert (
            self.genes.size
            == self.seqs.shape[0]
            == self.samples.size
            == self.Y.size
            == self.Z.size
        )

        self.seq_idx_embedder = utils.create_seq_idx_embedder()

    def __len__(self):
        return self.seqs.shape[0]

    def __getitem__(self, idx):
        seq = self.seq_idx_embedder[self.seqs[idx]]
        y = np.max(self.Y[idx], 0)
        return {"seq": seq, "y": y}


class SampleH5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str, seqlen: int, prefetch_seqs: bool = True):
        """
        If prefetch_seqs is True, then all sequences are loaded into memory. This makes initialization
        very slow (~15 minutes for the training set of h5_bins_384), but greatly speeds up __getitem__.
        We recommend setting this to False only if you are trying to debug.
        """
        super().__init__()
        assert seqlen % 128 == 0

        self.h5_file = h5py.File(h5_path, "r")
        self.seqlen = seqlen

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

    def get_total_n_bins(self):
        return self.seqlen // 128

    def __len__(self):
        return self.pairs.shape[0]

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
        seq = self.__shorten_seq(self.seqs[idx]).astype(np.float32)
        y = np.max(self.Y[idx], 0.0)
        return {"seq": seq, "y": y}


class SampleNormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str):
        data = np.load(filepath)
        self.genes = data["genes"]
        self.seqs = data["seqs"]
        self.samples = data["samples"]
        self.Y = data["Y"]
        self.Z = data["Z"]
        assert (
            self.genes.size
            == self.seqs.shape[0]
            == self.samples.size
            == self.Y.size
            == self.Z.size
        )

        self.seq_idx_embedder = utils.create_seq_idx_embedder()

    def get_total_n_bins(self):
        seqlen = self.seqs.shape[-1]
        assert seqlen % 128 == 0
        return seqlen // 128

    def __len__(self):
        return self.seqs.shape[0]

    def __getitem__(self, idx):
        seq = self.seq_idx_embedder[self.seqs[idx]]
        z = self.Z[idx]
        return {"seq": seq, "z": z}


class SampleNormalizedWithGeneDataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str):
        data = np.load(filepath)
        self.genes = data["genes"]
        self.seqs = data["seqs"]
        self.samples = data["samples"]
        self.Y = data["Y"]
        self.Z = data["Z"]
        assert (
            self.genes.size
            == self.seqs.shape[0]
            == self.samples.size
            == self.Y.size
            == self.Z.size
        )

        self.seq_idx_embedder = utils.create_seq_idx_embedder()
        self.unique_genes = np.unique(self.genes)
        self.gene_to_idx = {g: i for i, g in enumerate(np.unique(self.genes))}

    def get_total_n_bins(self):
        seqlen = self.seqs.shape[-1]
        assert seqlen % 128 == 0
        return seqlen // 128

    def __len__(self):
        return self.seqs.shape[0]

    def __getitem__(self, idx):
        seq = self.seq_idx_embedder[self.seqs[idx]]
        z = self.Z[idx]
        gene = self.genes[idx]
        return {"seq": seq, "z": z, "gene": gene}


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


class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str, n_pairs: int, half_precision=False):
        self.half_precision = half_precision

        data = np.load(filepath)
        self.genes = data["genes"]
        self.seqs = data["seqs"]
        self.samples = data["samples"]
        self.Y = data["Y"]
        self.Z = data["Z"]
        assert (
            self.genes.size
            == self.seqs.shape[0]
            == self.samples.size
            == self.Y.size
            == self.Z.size
        )

        self.pairs = self.__sample_pairs(n_pairs)
        self.seq_idx_embedder = utils.create_seq_idx_embedder()

    def __sample_pairs(self, n: int, random_seed: int = 42):
        unique_genes = np.unique(self.genes)
        gene_to_idxs = {g: np.where(self.genes == g)[0] for g in unique_genes}

        rng = np.random.default_rng(random_seed)
        sampled_genes = rng.choice(unique_genes, size=n, replace=True)

        pairs = [
            rng.choice(gene_to_idxs[g], size=2, replace=False) for g in sampled_genes
        ]
        return np.array(pairs)

    def get_total_n_bins(self):
        seqlen = self.seqs.shape[-1]
        assert seqlen % 128 == 0
        return seqlen // 128

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        assert self.genes[idx1] == self.genes[idx2]
        seq1 = self.seq_idx_embedder[self.seqs[idx1]]
        seq2 = self.seq_idx_embedder[self.seqs[idx2]]
        z_diff = self.Z[idx1] - self.Z[idx2]

        if self.half_precision:
            seq1 = torch.tensor(seq1).half()
            seq2 = torch.tensor(seq2).half()
            z_diff = torch.tensor(z_diff).half()

        return {"seq1": seq1, "seq2": seq2, "z_diff": z_diff}


class PairwiseDatasetIterable(torch.utils.data.IterableDataset):
    def __init__(self, filepath: str, random_seed: int = 42):
        super().__init__()

        data = np.load(filepath)
        self.genes = data["genes"]
        self.seqs = data["seqs"]
        self.samples = data["samples"]
        self.Y = data["Y"]
        self.Z = data["Z"]
        assert (
            self.genes.size
            == self.seqs.shape[0]
            == self.samples.size
            == self.Y.size
            == self.Z.size
        )

        self.seq_idx_embedder = utils.create_seq_idx_embedder()

        self.unique_genes = np.unique(self.genes)
        self.gene_to_idxs = {g: np.where(self.genes == g)[0] for g in self.unique_genes}

        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def __sample_pairs(self, n: int):
        sampled_genes = self.rng.choice(self.unique_genes, size=n, replace=True)

        pairs = [
            self.rng.choice(self.gene_to_idxs[g], size=2, replace=False)
            for g in sampled_genes
        ]
        return np.array(pairs)

    def get_total_n_bins(self):
        seqlen = self.seqs.shape[-1]
        assert seqlen % 128 == 0
        return seqlen // 128

    def worker_init_fn(self, worker_id):
        self.rng = np.random.default_rng(self.random_seed + worker_id)

    def __iter__(self):
        # randomly sample pairs
        sample_pair = self.__sample_pairs(1)[0]
        idx1, idx2 = sample_pair
        assert self.genes[idx1] == self.genes[idx2]
        seq1 = self.seq_idx_embedder[self.seqs[idx1]]
        seq2 = self.seq_idx_embedder[self.seqs[idx2]]
        z_diff = self.Z[idx1] - self.Z[idx2]
        yield {"seq1": seq1, "seq2": seq2, "z_diff": z_diff}


class PairwiseRegressionH5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: str,
        n_pairs_per_gene: int,
        seqlen: int,
        prefetch_seqs: bool = True,
        random_seed: int = 42,
    ):
        """
        If prefetch_seqs is True, then all sequences are loaded into memory. This makes initialization
        very slow (~15 minutes for the training set of h5_bins_384), but greatly speeds up __getitem__.
        We recommend setting this to False only if you are trying to debug.
        """
        super().__init__()
        assert seqlen % 128 == 0

        self.h5_file = h5py.File(h5_path, "r")
        self.n_pairs_per_gene = n_pairs_per_gene
        self.seqlen = seqlen
        self.rng = np.random.default_rng(random_seed)

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
        return self.pairs.shape[0]

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
        seq_idx1, seq_idx2 = self.pairs[idx]
        assert self.genes[seq_idx1] == self.genes[seq_idx2]
        assert self.samples[seq_idx1] != self.samples[seq_idx2]

        seq1 = self.__shorten_seq(self.seqs[seq_idx1].astype(np.float32))
        seq2 = self.__shorten_seq(self.seqs[seq_idx2].astype(np.float32))
        z_diff = self.Z[seq_idx1] - self.Z[seq_idx2]
        return {"seq1": seq1, "seq2": seq2, "z_diff": z_diff}


class PairwiseClassifcationH5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: str,
        n_pairs_per_gene: int,
        seqlen: int,
        min_percentile_diff: float = 25.0,
        prefetch_seqs: bool = True,
        random_seed: int = 42,
    ):
        """
        If prefetch_seqs is True, then all sequences are loaded into memory. This makes initialization
        very slow (~15 minutes for the training set of h5_bins_384), but greatly speeds up __getitem__.
        We recommend setting this to False only if you are trying to debug.
        """
        super().__init__()
        assert seqlen % 128 == 0

        self.h5_file = h5py.File(h5_path, "r")
        self.n_pairs_per_gene = n_pairs_per_gene
        self.seqlen = seqlen
        self.min_percentile_diff = min_percentile_diff
        self.rng = np.random.default_rng(random_seed)

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
        return self.pairs.shape[0]

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
        seq_idx1, seq_idx2 = self.pairs[idx]
        assert self.genes[seq_idx1] == self.genes[seq_idx2]
        assert self.samples[seq_idx1] != self.samples[seq_idx2]

        seq1 = self.__shorten_seq(self.seqs[seq_idx1].astype(np.float32))
        seq2 = self.__shorten_seq(self.seqs[seq_idx2].astype(np.float32))
        Y = int(self.Y[seq_idx1] >= self.Y[seq_idx2])
        return {"seq1": seq1, "seq2": seq2, "Y": Y}


class PairwiseMPRADataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str, split: str, reverse_complement: bool = False):
        super().__init__()

        assert split in [
            "train",
            "val",
            "test",
        ], "split must be one of train, val, test"

        self.split = split

        self.data = pd.read_csv(filepath, sep="\t")
        self.data = self.data[self.data[f"is_{split}"]].reset_index(drop=True)

        self.ref_sequences = self.data["ref_sequence"].values
        self.alt_sequences = self.data["alt_sequence"].values
        self.cell_types = []
        self.variant_effects = []
        for col in self.data.columns:
            if "_normalized_variant_effect" in col:
                self.cell_types.append(col.split("_")[0])
                self.variant_effects.append(self.data[col].values)
        self.variant_effects = np.stack(self.variant_effects, axis=1)
        assert self.variant_effects.shape[1] == len(self.cell_types)
        assert self.variant_effects.shape[0] == len(self.ref_sequences)

        self.seq_idx_embedder = utils.create_seq_idx_embedder()
        self.ref_sequences = np.array(
            [self.seq_idx_embedder[[ord(s) for s in seq]] for seq in self.ref_sequences]
        )
        self.alt_sequences = np.array(
            [self.seq_idx_embedder[[ord(s) for s in seq]] for seq in self.alt_sequences]
        )
        assert (
            self.ref_sequences.shape[0]
            == self.alt_sequences.shape[0]
            == self.variant_effects.shape[0]
        )
        assert self.ref_sequences.shape[1] == self.alt_sequences.shape[1] == 200

        self.mask = ~np.isnan(self.variant_effects)
        assert self.mask.shape[0] == self.variant_effects.shape[0]
        assert self.mask.shape[1] == self.variant_effects.shape[1]

        self.reverse_complement = reverse_complement
        if self.split == "val" or self.split == "test":
            if self.reverse_complement:
                raise ValueError(
                    "reverse_complement must be False for val and test splits"
                )
        else:
            if not self.reverse_complement:
                print(
                    "WARNING: reverse_complement and random_shift are both False for train split. Setting these to True can improve model performance."
                )

    def get_num_cells(self):
        return self.variant_effects.shape[1]

    def get_cell_names(self):
        return self.cell_types

    def get_total_n_bins(self):
        seqlen = len(self.ref_sequences[0])
        return int(np.ceil(seqlen / 128))

    def __len__(self):
        return len(self.ref_sequences)

    def __getitem__(self, idx):
        ref_seq = self.ref_sequences[idx]
        alt_seq = self.alt_sequences[idx]
        variant_effect = self.variant_effects[idx]
        mask = self.mask[idx]

        # one-hot encode the sequences
        ref_seq = seq_indices_to_one_hot(torch.tensor(ref_seq)).detach().numpy()
        alt_seq = seq_indices_to_one_hot(torch.tensor(alt_seq)).detach().numpy()

        if self.reverse_complement:
            coin_flip = np.random.choice([True, False])
            if coin_flip:
                ref_seq = np.flip(ref_seq, axis=0)
                alt_seq = np.flip(alt_seq, axis=0)

                # order of bases is ACGT, so reverse-complement is just flipping the sequence
                ref_seq = np.flip(ref_seq, axis=1)
                alt_seq = np.flip(alt_seq, axis=1)

        ref_seq = torch.tensor(ref_seq.copy()).float()
        alt_seq = torch.tensor(alt_seq.copy()).float()
        variant_effect = torch.tensor(variant_effect.copy()).float()
        mask = torch.tensor(mask.copy()).bool()

        return {
            "ref_seq": ref_seq,
            "alt_seq": alt_seq,
            "variant_effect": variant_effect,
            "mask": mask,
        }


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
