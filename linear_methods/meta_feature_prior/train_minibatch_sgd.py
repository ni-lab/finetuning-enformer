import os
from argparse import ArgumentParser

import h5py

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path

        self.genes = []
        self.n_variants_per_gene = {}  # gene -> int
        self.samples_per_gene = {}  # gene -> (n_samples,)
        self.X_per_gene = {}  # gene -> (n_samples, n_variants)
        self.F_per_gene = {}  # gene -> (n_variants, n_meta_features)
        self.Y_per_gene = {}  # gene -> (n_samples,)

        with h5py.File(h5_path, "r") as f:
            for g in tqdm(f["genes"][:].astype(str), desc="Loading gene data"):
                g_prime = g.replace(".", "_")  # module name cannot contain "."
                self.genes.append(g_prime)
                self.n_variants_per_gene[g_prime] = f[g]["variants"].shape[0]
                self.samples_per_gene[g_prime] = f[g]["samples"][:].astype(str)
                self.X_per_gene[g_prime] = f[g]["dosages"][:].astype(np.float16)
                self.F_per_gene[g_prime] = f[g]["meta_features"][:].astype(np.float16)
                self.Y_per_gene[g_prime] = f[g]["z_scores"][:].astype(np.float16)

        self.max_n_variants = max(self.n_variants_per_gene.values())
        self.n_samples = len(self.samples_per_gene[self.genes[0]])

        # Collate data into large matrices
        X_all_shape = (len(self.genes), self.n_samples, self.max_n_variants)
        F_all_shape = (len(self.genes), self.max_n_variants, self.get_n_meta_features())
        Y_all_shape = (len(self.genes), self.n_samples)
        mask_shape = (len(self.genes), self.max_n_variants)
        print(f"X_all_shape: {X_all_shape}")
        print(f"F_all_shape: {F_all_shape}")
        print(f"Y_all_shape: {Y_all_shape}")
        print(f"mask_shape: {mask_shape}")

        self.X_all = np.zeros(X_all_shape, dtype=np.float16)
        self.F_all = np.zeros(F_all_shape, dtype=np.float16)
        self.Y_all = np.zeros(Y_all_shape, dtype=np.float16)
        self.mask_all = np.zeros(mask_shape, dtype=np.float16)
        for i, g in enumerate(self.genes):
            n_variants = self.get_n_variants_before_padding(g)
            self.X_all[i, :, :n_variants] = self.X_per_gene[g]
            self.F_all[i, :n_variants, :] = self.F_per_gene[g]
            self.Y_all[i, :] = self.Y_per_gene[g]
            self.mask_all[i, :n_variants] = 1

        assert not np.isnan(self.X_all).any()
        assert not np.isnan(self.F_all).any()
        assert not np.isnan(self.Y_all).any()
        assert not np.isnan(self.mask_all).any()

    def get_n_variants_after_padding(self):
        return self.max_n_variants

    def get_n_variants_before_padding(self, gene: str):
        return self.n_variants_per_gene[gene]

    def get_n_meta_features(self):
        n_meta_features = [self.F_per_gene[g].shape[1] for g in self.genes]
        assert all(n == n_meta_features[0] for n in n_meta_features)
        return n_meta_features[0]

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        return {
            "gene": self.genes[idx],
            "X": self.X_all[idx].astype(np.float32),
            "F": self.F_all[idx].astype(np.float32),
            "Y": self.Y_all[idx].astype(np.float32),
            "mask": self.mask_all[idx].astype(np.float32),
        }


class Model(L.LightningModule):
    def __init__(
        self,
        genes: list[str],
        n_variants: int,
        n_meta_features: int,
        C: float,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.genes = genes
        self.gene_to_idx = {g: i for i, g in enumerate(genes)}
        self.C = C

        # Initialize a learnable parameter W of size (n_genes, n_variants)
        self.W = nn.Parameter(torch.empty(len(genes), n_variants))
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(torch.zeros(len(genes)))

        # We have a single linear model that maps meta-features to priors on variant effects
        self.prior = nn.Linear(n_meta_features, 1)
        self.softplus = nn.Softplus()

        self.mse_loss = nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def compute_prior(self, F):
        """
        Parameters:
            F: (n_genes, n_variants, n_meta_features)
        Returns:
            prior: (n_genes, n_variants)
        """
        out = self.softplus(self.prior(F))  # (n_genes, n_variants, 1)
        return out.squeeze(-1)  # (n_genes, n_variants)

    def forward(self, genes, X):
        """
        Parameters:
            genes: (n_genes)
            X: (n_genes, n_samples, n_variants)
        Returns:
            preds: (n_genes, n_samples)
        """
        gene_idxs = [self.gene_to_idx[g] for g in genes]
        my_W = self.W[gene_idxs]  # (n_genes, n_variants)
        my_b = self.b[gene_idxs]  # (n_genes)

        preds = torch.einsum("ijk,ik->ij", X, my_W)  # (n_genes, n_samples)
        preds += my_b.unsqueeze(1)  # (n_genes, n_samples)
        return preds

    def training_step(self, batch, batch_idx):
        genes = batch["gene"]  # (n_genes)
        X = batch["X"]  # (n_genes, n_samples, n_variants)
        F = batch["F"]  # (n_genes, n_variants, n_meta_features)
        Y = batch["Y"]  # (n_genes, n_samples)
        mask = batch["mask"]  # (n_genes, n_variants)

        priors = self.compute_prior(F)  # (n_genes, n_variants)
        Y_hat = self(genes, X)  # (n_genes, n_samples)

        # Compute prediction loss
        pred_loss = self.mse_loss(Y_hat, Y)
        self.log("train/pred_loss", pred_loss)

        # Compute regularization loss
        gene_idxs = [self.gene_to_idx[g] for g in genes]
        my_W = self.W[gene_idxs]  # (n_genes, n_variants)
        reg_mtx = mask * (my_W**2) / (priors)
        regularization_loss = reg_mtx.sum()
        self.log("train/regularization_loss", regularization_loss)

        # Compute prior loss
        prior_loss = self.C * (mask * priors).sum()
        self.log("train/prior_loss", prior_loss)

        total_loss = pred_loss + regularization_loss + prior_loss
        self.log("train/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        genes = batch["gene"]  # (n_genes)
        X = batch["X"]  # (n_genes, n_samples, n_variants)
        Y = batch["Y"]  # (n_genes, n_samples)

        Y_hat = self(genes, X)  # (n_genes, n_samples)
        pred_loss = self.mse_loss(Y_hat, Y)
        self.log("val/pred_loss", pred_loss)
        return pred_loss


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("C", type=float)
    parser.add_argument("run_name", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    train_ds = Dataset(os.path.join(args.dataset_dir, "train.h5"))
    val_ds = Dataset(os.path.join(args.dataset_dir, "val.h5"))
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=4
    )

    run_dir = os.path.join(args.save_dir, args.run_name + f"_C_{args.C}_lr_{args.lr}")
    logs_dir = os.path.join(run_dir, "logs")
    ckpts_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpts_dir, exist_ok=True)

    logger = WandbLogger(
        project="meta_feature_prior", name=args.run_name, save_dir=logs_dir
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpts_dir,
        monitor="val/pred_loss",
        mode="min",
        save_top_k=3,
        auto_insert_metric_name=False,
    )
    early_stopping_cb = EarlyStopping(
        monitor="val/pred_loss",
        mode="min",
        patience=5,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
        max_epochs=1_000,
        logger=logger,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    model = Model(
        genes=train_ds.genes,
        n_variants=train_ds.get_n_variants_after_padding(),
        n_meta_features=train_ds.get_n_meta_features(),
        C=args.C,
        lr=args.lr,
    )

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
