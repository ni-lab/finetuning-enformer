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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path

        self.genes = []
        self.samples_per_gene = {}  # gene -> (n_samples,)
        self.X_per_gene = {}  # gene -> (n_samples, n_variants)
        self.F_per_gene = {}  # gene -> (n_variants, n_meta_features)
        self.Y_per_gene = {}  # gene -> (n_samples,)
        with h5py.File(h5_path, "r") as f:
            self.max_n_variants = max(f[g]["variants"].shape[1] for g in f["genes"])
            for g in sorted(f["genes"][:].astype(str)):
                g_prime = g.replace(".", "_")  # module name cannot contain "."
                self.genes.append(g_prime)
                self.samples_per_gene[g_prime] = f[g]["samples"][:].astype(str)

                # Pad features with zeros
                X = f[g]["dosages"][:].astype(np.float32)
                X_pad = np.zeros(
                    (X.shape[0], self.max_n_variants - X.shape[1]), dtype=np.float32
                )
                self.X_per_gene[g_prime] = np.hstack([X, X_pad])

                # Pad meta-features with zeros
                F = f[g]["meta_features"][:].astype(np.float32)
                F_pad = np.zeros(
                    (self.max_n_variants - F.shape[0], F.shape[1]), dtype=np.float32
                )
                self.F_per_gene[g_prime] = np.vstack([F, F_pad])

                self.Y_per_gene[g_prime] = f[g]["z_scores"][:].astype(np.float32)

        for g in self.genes:
            assert (
                self.samples_per_gene[g].shape[0]
                == self.X_per_gene[g].shape[0]
                == self.Y_per_gene[g].shape[0]
            )
            assert self.X_per_gene[g].shape[1] == self.F_per_gene[g].shape[0]
            assert not np.isnan(self.X_per_gene[g]).any()
            assert not np.isnan(self.F_per_gene[g]).any()
            assert not np.isnan(self.Y_per_gene[g]).any()

    def get_n_variants_after_padding(self):
        return self.max_n_variants

    def get_n_variants_before_padding(self):
        variants_per_gene = {}
        with h5py.File(self.h5_path, "r") as f:
            for g in f["genes"].astype(str):
                variants_per_gene[g] = f[g]["variants"].shape[1]
        return variants_per_gene

    def get_n_meta_features(self):
        n_meta_features = [self.F_per_gene[g].shape[1] for g in self.genes]
        assert all(n == n_meta_features[0] for n in n_meta_features)
        return n_meta_features[0]

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        gene = self.genes[idx]
        return {
            "gene": gene,
            "X": self.X_per_gene[gene],
            "F": self.F_per_gene[gene],
            "Y": self.Y_per_gene[gene],
        }


class Model(L.LightningModule):
    def __init__(
        self,
        genes: list[str],
        n_varants: int,
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
        self.W = nn.Parameter(torch.empty(len(genes), n_varants))
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
        return self.softplus(self.prior(F))

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

        priors = self.compute_prior(F)  # (n_genes, n_variants)

    def training_step(self, batch, batch_idx):
        gene = batch["gene"][0]
        features = batch["feature"][0]  # (n_samples, n_variants)
        meta_features = batch["meta_feature"][0]  # (n_variants, n_meta_features)
        Y = batch["Y"][0]  # (n_samples)
        assert features.shape[0] == Y.shape[0]
        assert features.shape[1] == meta_features.shape[0]

        priors = self.compute_prior(meta_features)  # (n_variants)
        assert not priors.isnan().any()

        preds = self(gene, features)  # (n_samples)

        # Compute prediction loss
        pred_loss = self.mse_loss(preds, Y)
        self.log("train/pred_loss", pred_loss)

        # Compute regularization loss: sum(betas^2 / priors)
        regularization_loss = torch.div(
            torch.square(self.betas[gene].weight.squeeze()), priors + 1e-3
        ).sum()
        self.log("train/regularization_loss", regularization_loss)

        # Compute prior loss
        prior_loss = self.C * priors.sum()
        self.log("train/prior_loss", prior_loss)

        loss = pred_loss + regularization_loss + prior_loss
        self.log("train/total_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        gene = batch["gene"][0]
        features = batch["feature"][0]  # (n_samples, n_variants)
        meta_features = batch["meta_feature"][0]  # (n_variants, n_meta_features)
        Y = batch["Y"][0]  # (n_samples)
        assert features.shape[0] == Y.shape[0]
        assert features.shape[1] == meta_features.shape[0]

        preds = self(gene, features)  # (n_samples)
        pred_loss = self.mse_loss(preds, Y)
        self.log("val/pred_loss", pred_loss)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("C", type=float)
    parser.add_argument("run_name", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--lr", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()

    train_ds = Dataset(os.path.join(args.dataset_dir, "train.h5"))
    val_ds = Dataset(os.path.join(args.dataset_dir, "val.h5"))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1)

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
        max_epochs=1_000,
        logger=logger,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    model = Model(
        genes=train_ds.genes,
        n_variants_per_gene=train_ds.get_n_variants_per_gene(),
        n_meta_features=train_ds.get_n_meta_features(),
        C=args.C,
        lr=args.lr,
    )

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
