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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str, load_meta_features: bool = True):
        self.genes = []
        self.n_variants_per_gene = {}  # gene -> int
        self.samples_per_gene = {}  # gene -> (n_samples,)
        self.X_per_gene = {}  # gene -> (n_samples, n_variants)
        self.F_per_gene = {}  # gene -> (n_variants, n_meta_features)
        self.Y_per_gene = {}  # gene -> (n_samples,)

        with h5py.File(h5_path, "r") as f:
            for g in tqdm(f["genes"][:].astype(str), desc="Loading gene data"):
                g_prime = g.replace(".", "_")
                self.genes.append(g_prime)
                self.n_variants_per_gene[g_prime] = f[g]["variants"].shape[0]
                self.samples_per_gene[g_prime] = f[g]["samples"][:].astype(str)
                self.X_per_gene[g_prime] = f[g]["dosages"][:].astype(np.float32)
                self.Y_per_gene[g_prime] = f[g]["z_scores"][:].astype(np.float32)
                if load_meta_features:
                    self.F_per_gene[g_prime] = f[g]["meta_features"][:].astype(
                        np.float32
                    )

        for g in self.genes:
            assert not np.isnan(self.X_per_gene[g]).any()
            assert not np.isnan(self.Y_per_gene[g]).any()
            if load_meta_features:
                assert not np.isnan(self.F_per_gene[g]).any()

    def get_n_variants_per_gene(self):
        return self.n_variants_per_gene

    def get_n_meta_features(self):
        if len(self.F_per_gene) == 0:
            raise ValueError("No meta-features loaded")

        n_meta_features = [self.F_per_gene[g].shape[1] for g in self.genes]
        assert all(n == n_meta_features[0] for n in n_meta_features)
        return n_meta_features[0]

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        g = self.genes[idx]
        result = dict(gene=g, X=self.X_per_gene[g], Y=self.Y_per_gene[g])
        if len(self.F_per_gene) > 0:
            result["F"] = self.F_per_gene[g]
        return result


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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=1, threshold=5e-4
            ),
            "monitor": "val/pred_loss",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def compute_prior(self, F):
        """
        Parameters:
            F: (n_variants, n_meta_features)
        Returns:
            prior: (n_variants)
        """
        return self.softplus(self.prior(F)).squeeze()

    def forward(self, gene, X):
        """
        Parameters:
            gene: str
            X: (n_samples, n_variants)
        Returns:
            preds: (n_samples)
        """
        return self.betas[gene](X).squeeze()

    def training_step(self, batch, batch_idx):
        gene = batch["gene"][0]
        X = batch["X"][0]  # (n_samples, n_variants)
        F = batch["F"][0]  # (n_variants, n_meta_features)
        Y = batch["Y"][0]  # (n_samples)
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == F.shape[0]

        priors = self.compute_prior(F)  # (n_variants)
        assert not priors.isnan().any()
        Y_hat = self(gene, X)  # (n_samples)

        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        # Compute prediction loss
        pred_loss = self.mse_loss(Y_hat, Y)
        self.log("train/pred_loss", pred_loss)

        # Compute regularization loss
        gene_beta = self.betas[gene].weight.squeeze()  # (n_variants)
        reg_loss = ((gene_beta) ** 2 / (priors + 1e-3)).sum()
        self.log("train/regularization_loss", reg_loss)

        # Compute prior loss
        prior_loss = self.C * priors.sum()
        self.log("train/prior_loss", prior_loss)

        total_loss = pred_loss + reg_loss + prior_loss
        self.log("train/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        gene = batch["gene"][0]
        X = batch["X"][0]  # (n_samples, n_variants)
        Y = batch["Y"][0]  # (n_samples)
        assert X.shape[0] == Y.shape[0]

        Y_hat = self(gene, X)  # (n_samples)
        pred_loss = self.mse_loss(Y_hat, Y)
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
    val_ds = Dataset(os.path.join(args.dataset_dir, "val.h5"), load_meta_features=False)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=1, num_workers=2, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=1, num_workers=2, shuffle=False
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
        log_every_n_steps=100,
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
