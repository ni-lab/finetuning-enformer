import pdb

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from enformer_pytorch import Enformer as BaseEnformer
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class AttentionPool(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, X):
        """
        X shape: (..., tokens, embedding_dim)
        """
        attn_scores = self.attention(X)  # (..., tokens, 1)
        attn_weights = F.softmax(attn_scores, dim=-2)  # (..., tokens, 1)
        output = (attn_weights * X).sum(dim=-2)  # (..., embedding_dim)
        return output


class PairwiseFinetuned(L.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        n_total_bins: int,
        avg_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if checkpoint is None:
            self.base = BaseEnformer.from_pretrained(
                "EleutherAI/enformer-official-rough", target_length=n_total_bins
            )
        else:
            checkpoint = torch.load(checkpoint)
            new_state_dict = {}
            # if Enformer is component of a larger model, we can subset the state dict of the larger model using the state_dict_subset_prefix
            # for the model fine-tuned on MPRA data, prefix is "model.Backbone.model."
            if state_dict_subset_prefix is not None:
                print(
                    "Loading subset of state dict from checkpoint, key prefix: ",
                    state_dict_subset_prefix,
                )
                for key in checkpoint["state_dict"]:
                    if key.startswith(state_dict_subset_prefix):
                        new_state_dict[
                            key[len(state_dict_subset_prefix) :]
                        ] = checkpoint["state_dict"][key]
            else:
                new_state_dict = checkpoint["state_dict"]
            self.base = BaseEnformer.from_pretrained(
                "EleutherAI/enformer-official-rough", target_length=n_total_bins
            )
            self.base.load_state_dict(new_state_dict)

        enformer_hidden_dim = 2 * self.base.dim
        self.attention_pool = AttentionPool(enformer_hidden_dim)
        self.prediction_head = nn.Linear(enformer_hidden_dim, 1)
        self.mse_loss = nn.MSELoss()

        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins

    def forward(self, X):
        """
        X (tensor): (sample, haplotype, length)
        """
        X = rearrange(X, "S H L -> (S H) L")
        X = self.base(
            X, return_only_embeddings=True
        )  # (S * H, n_total_bins, enformer_hidden_dim)

        assert X.shape[1] == self.hparams.n_total_bins
        X = X[:, self.center_start : self.center_end, :]
        X = self.attention_pool(X)  # (S * H, enformer_hidden_dim)
        Y = self.prediction_head(X)  # (S * H, 1)
        Y = rearrange(Y, "(S H) 1 -> S H", H=2)
        Y = Y.mean(dim=1)
        return Y

    def get_mse_loss(self, X1, X2, Y):
        """
        X1 (tensor): (sample, haplotype, length)
        X2 (tensor): (sample, haplotype, length)
        Y (tensor): (sample,)
        """
        X = torch.cat([X1, X2], dim=0)
        Y_hat = self(X)
        Y_hat = Y_hat[: X1.shape[0]] - Y_hat[X1.shape[0] :]
        return self.mse_loss(Y_hat, Y)

    def training_step(self, batch, batch_idx):
        X1, X2, Y = batch["seq1"], batch["seq2"], batch["z_diff"].float()
        mse_loss = self.get_mse_loss(X1, X2, Y)
        self.log("train/mse_loss", mse_loss)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log(
            "train/weight_decay",
            self.trainer.optimizers[0].param_groups[0]["weight_decay"],
        )
        return mse_loss

    def validation_step(self, batch, batch_idx):
        X1, X2, Y = batch["seq1"], batch["seq2"], batch["z_diff"].float()
        mse_loss = self.get_mse_loss(X1, X2, Y)
        self.log("val/mse_loss", mse_loss)
        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=10000,
            max_epochs=self.trainer.max_steps,
            eta_min=self.hparams.lr / 100,
        )

        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return config


class SingleFinetuned(L.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        n_total_bins: int,
        avg_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if checkpoint is None:
            self.base = BaseEnformer.from_pretrained(
                "EleutherAI/enformer-official-rough", target_length=n_total_bins
            )
        else:
            checkpoint = torch.load(checkpoint)
            new_state_dict = {}
            # if Enformer is component of a larger model, we can subset the state dict of the larger model using the state_dict_subset_prefix
            # for the model fine-tuned on MPRA data, prefix is "model.Backbone.model."
            if state_dict_subset_prefix is not None:
                print(
                    "Loading subset of state dict from checkpoint, key prefix: ",
                    state_dict_subset_prefix,
                )
                for key in checkpoint["state_dict"]:
                    if key.startswith(state_dict_subset_prefix):
                        new_state_dict[
                            key[len(state_dict_subset_prefix) :]
                        ] = checkpoint["state_dict"][key]
            else:
                new_state_dict = checkpoint["state_dict"]
            self.base = BaseEnformer.from_pretrained(
                "EleutherAI/enformer-official-rough", target_length=n_total_bins
            )
            self.base.load_state_dict(new_state_dict)

        enformer_hidden_dim = 2 * self.base.dim
        self.attention_pool = AttentionPool(enformer_hidden_dim)
        self.prediction_head = nn.Linear(enformer_hidden_dim, 1)
        self.mse_loss = nn.MSELoss()

        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins

    def forward(self, X):
        """
        X (tensor): (sample, haplotype, length)
        """
        X = rearrange(X, "S H L -> (S H) L")
        X = self.base(
            X, return_only_embeddings=True
        )  # (S * H, n_total_bins, enformer_hidden_dim)

        assert X.shape[1] == self.hparams.n_total_bins
        X = X[:, self.center_start : self.center_end, :]
        X = self.attention_pool(X)  # (S * H, enformer_hidden_dim)
        Y = self.prediction_head(X)  # (S * H, 1)
        Y = rearrange(Y, "(S H) 1 -> S H", H=2)
        Y = Y.mean(dim=1)
        return Y

    def training_step(self, batch, batch_idx):
        X, Y = batch["seq"], batch["z"].float()
        Y_hat = self(X)
        mse_loss = self.mse_loss(Y_hat, Y)
        self.log("train/mse_loss", mse_loss)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log(
            "train/weight_decay",
            self.trainer.optimizers[0].param_groups[0]["weight_decay"],
        )
        return mse_loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch["seq"], batch["z"].float()
        Y_hat = self(X)
        mse_loss = self.mse_loss(Y_hat, Y)
        self.log("val/mse_loss", mse_loss)
        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=10000,
            max_epochs=self.trainer.max_steps,
            eta_min=self.hparams.lr / 100,
        )

        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return config


class SingleWithGeneFinetuned(L.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        all_genes: list,
        n_total_bins: int,
        avg_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if checkpoint is None:
            self.base = BaseEnformer.from_pretrained(
                "EleutherAI/enformer-official-rough", target_length=n_total_bins
            )
        else:
            checkpoint = torch.load(checkpoint)
            new_state_dict = {}
            # if Enformer is component of a larger model, we can subset the state dict of the larger model using the state_dict_subset_prefix
            # for the model fine-tuned on MPRA data, prefix is "model.Backbone.model."
            if state_dict_subset_prefix is not None:
                print(
                    "Loading subset of state dict from checkpoint, key prefix: ",
                    state_dict_subset_prefix,
                )
                for key in checkpoint["state_dict"]:
                    if key.startswith(state_dict_subset_prefix):
                        new_state_dict[
                            key[len(state_dict_subset_prefix) :]
                        ] = checkpoint["state_dict"][key]
            else:
                new_state_dict = checkpoint["state_dict"]
            self.base = BaseEnformer.from_pretrained(
                "EleutherAI/enformer-official-rough", target_length=n_total_bins
            )
            self.base.load_state_dict(new_state_dict)

        enformer_hidden_dim = 2 * self.base.dim
        self.all_genes = all_genes
        num_genes = len(all_genes)
        self.gene_to_id = {gene: i for i, gene in enumerate(all_genes)}
        self.attention_pool = AttentionPool(enformer_hidden_dim)
        # have an embedding for each gene
        self.gene_embedding = nn.Embedding(num_genes, enformer_hidden_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(enformer_hidden_dim * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.mse_loss = nn.MSELoss()

        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins

    def forward(self, X, genes):
        """
        X (tensor): (sample, haplotype, length)
        gene (tensor): (sample,)
        """
        gene_ids = torch.tensor([self.gene_to_id[g] for g in genes], device=X.device)
        X = rearrange(X, "S H L -> (S H) L")
        X = self.base(
            X, return_only_embeddings=True
        )  # (S * H, n_total_bins, enformer_hidden_dim)

        assert X.shape[1] == self.hparams.n_total_bins
        X = X[:, self.center_start : self.center_end, :]
        X = self.attention_pool(X)  # (S * H, enformer_hidden_dim)
        gene_embedding = self.gene_embedding(gene_ids)  # (S, enformer_hidden_dim)
        gene_embedding = (
            gene_embedding.unsqueeze(1).repeat(1, 2, 1).flatten(0, 1)
        )  # (S * H, enformer_hidden_dim)
        X = torch.cat([X, gene_embedding], dim=-1)  # (S * H, enformer_hidden_dim * 2)
        Y = self.prediction_head(X)  # (S * H, 1)
        Y = rearrange(Y, "(S H) 1 -> S H", H=2)
        Y = Y.mean(dim=1)
        return Y

    def training_step(self, batch, batch_idx):
        X, Y, genes = batch["seq"], batch["z"].float(), batch["gene"]
        Y_hat = self(X, genes)
        mse_loss = self.mse_loss(Y_hat, Y)
        self.log("train/mse_loss", mse_loss)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log(
            "train/weight_decay",
            self.trainer.optimizers[0].param_groups[0]["weight_decay"],
        )
        return torch.exp(mse_loss)

    def validation_step(self, batch, batch_idx):
        X, Y, genes = batch["seq"], batch["z"].float(), batch["gene"]
        Y_hat = self(X, genes)
        mse_loss = self.mse_loss(Y_hat, Y)
        self.log("val/mse_loss", mse_loss)
        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=10000,
            max_epochs=self.trainer.max_steps,
            eta_min=self.hparams.lr / 100,
        )

        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return config
