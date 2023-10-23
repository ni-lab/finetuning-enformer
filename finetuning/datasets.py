import numpy as np
import torch
import utils


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
        y = np.min(self.Y[idx], 0)
        return {"seq": seq, "y": y}


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
    def __init__(self, filepath: str, n_pairs: int):
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
        return {"seq1": seq1, "seq2": seq2, "z_diff": z_diff}
