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
        self.genes = []
        self.samples = {}  # gene -> (n_samples,)
        self.features = {}  # gene -> (n_samples, n_variants)
        self.meta_features = {}  # gene -> (n_variants, n_meta_features)
        self.Y = {}  # gene -> (n_samples,)
        with h5py.File(h5_path, "r") as f:
            for g in sorted(f["genes"][:].astype(str)):
                if f[g]["variants"].shape[0] == 0:
                    continue
                g_renamed = g.replace(".", "_")  # module name cannot contain "."
                self.genes.append(g_renamed)
                self.samples[g_renamed] = f[g]["samples"][:].astype(str)
                self.features[g_renamed] = f[g]["dosages"][:].astype(np.float32)
                self.meta_features[g_renamed] = f[g]["meta_features"][:].astype(
                    np.float32
                )
                self.Y[g_renamed] = f[g]["z_scores"][:].astype(np.float32)

        for g in self.genes:
            assert (
                self.samples[g].shape[0]
                == self.features[g].shape[0]
                == self.Y[g].shape[0]
            )
            assert self.features[g].shape[1] == self.meta_features[g].shape[0]
            assert not np.isnan(self.features[g]).any()
            assert not np.isnan(self.meta_features[g]).any(), g
            assert not np.isnan(self.Y[g]).any()

    def get_n_variants_per_gene(self):
        return {g: self.features[g].shape[1] for g in self.genes}

    def get_n_meta_features(self):
        n_meta_features = [self.meta_features[g].shape[1] for g in self.genes]
        assert all(n == n_meta_features[0] for n in n_meta_features)
        return n_meta_features[0]

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        gene = self.genes[idx]
        return {
            "gene": gene,
            "feature": self.features[gene],
            "meta_feature": self.meta_features[gene],
            "Y": self.Y[gene],
        }


class Model(L.LightningModule):
    def __init__(
        self,
        genes: list[str],
        n_variants_per_gene: dict[str, int],
        n_meta_features: int,
        C: float,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.genes = genes
        self.gene_to_idx = {g: i for i, g in enumerate(genes)}
        self.C = C

        # For each gene, we have a linear model with n_variants_per_gene[gene] features
        self.betas = nn.ModuleDict(
            {gene: nn.Linear(n_variants_per_gene[gene], 1) for gene in genes}
        )

        # We have a single linear model that maps meta-features to priors on variant effects
        self.prior = nn.Linear(n_meta_features, 1)
        self.softplus = nn.Softplus()

        self.mse_loss = nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def compute_prior(self, meta_features):
        """
        Parameters:
            meta_features: (n_variants, n_meta_features)
        Returns:
            prior: (n_variants)
        """
        return self.softplus(self.prior(meta_features)).squeeze()

    def forward(self, gene, features):
        """
        Parameters:
            gene: str
            features: (n_samples, n_variants)
        Returns:
            preds: (n_samples)
        """
        return self.betas[gene](features).squeeze()

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
