import pdb

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from einops import rearrange
from enformer_pytorch import Enformer as BaseEnformer
from enformer_pytorch.data import seq_indices_to_one_hot, str_to_one_hot
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from torchmetrics import AUROC, Accuracy, Precision, R2Score, Recall


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


class PairwiseFinetunedMPRA(L.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        n_total_bins: int,
        num_cells: int,
        cell_names: list,
        checkpoint=None,
        state_dict_subset_prefix=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cell_names = cell_names
        self.num_cells = num_cells

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
        self.prediction_head = nn.Linear(enformer_hidden_dim * n_total_bins, num_cells)
        self.mse_loss = nn.MSELoss()

    def forward(self, X):
        """
        X (tensor): (sample, length)
        """
        X = self.base(
            X, return_only_embeddings=True
        )  # (S, n_total_bins, enformer_hidden_dim)

        assert X.shape[1] == self.hparams.n_total_bins
        X = rearrange(X, "S L enformer_hidden_dim -> S (L enformer_hidden_dim)")
        Y = self.prediction_head(X)  # (S, num_cells)
        return Y

    def training_step(self, batch, batch_idx):
        ref_seq, alt_seq, variant_effect, mask = (
            batch["ref_seq"],
            batch["alt_seq"],
            batch["variant_effect"].float(),
            batch["mask"],
        )
        ref_seq_pred = self(ref_seq)
        alt_seq_pred = self(alt_seq)
        variant_effect_pred = alt_seq_pred - ref_seq_pred
        mse_loss = self.mse_loss(variant_effect_pred[mask], variant_effect[mask])

        self.log("train/mse_loss", mse_loss)
        # also log per cell type mse loss
        for i, cell_name in enumerate(self.cell_names):
            mask_cell = mask[:, i]
            if mask_cell.sum() == 0:
                continue
            mse_loss_cell = self.mse_loss(
                variant_effect_pred[mask_cell, i], variant_effect[mask_cell, i]
            )
            self.log(f"train/mse_loss_{cell_name}", mse_loss_cell)

        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log(
            "train/weight_decay",
            self.trainer.optimizers[0].param_groups[0]["weight_decay"],
        )
        return mse_loss

    def validation_step(self, batch, batch_idx):
        ref_seq, alt_seq, variant_effect, mask = (
            batch["ref_seq"],
            batch["alt_seq"],
            batch["variant_effect"].float(),
            batch["mask"],
        )
        ref_seq_pred = self(ref_seq)
        alt_seq_pred = self(alt_seq)
        variant_effect_pred = alt_seq_pred - ref_seq_pred
        mse_loss = self.mse_loss(variant_effect_pred[mask], variant_effect[mask])

        self.log("val/mse_loss", mse_loss)
        # also log per cell type mse loss
        for i, cell_name in enumerate(self.cell_names):
            mask_cell = mask[:, i]
            if mask_cell.sum() == 0:
                continue
            mse_loss_cell = self.mse_loss(
                variant_effect_pred[mask_cell, i], variant_effect[mask_cell, i]
            )
            self.log(f"val/mse_loss_{cell_name}", mse_loss_cell)

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


class PairwiseWithOriginalDataJointTrainingFloatPrecision(L.LightningModule):
    def __init__(
        self,
        lr: float,
        n_total_bins: int,
        avg_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if checkpoint is None:
            self.base = BaseEnformer.from_pretrained(
                "EleutherAI/enformer-official-rough"
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
                "EleutherAI/enformer-official-rough"
            )
            self.base.load_state_dict(new_state_dict)

        enformer_hidden_dim = 2 * self.base.dim
        self.attention_pool = AttentionPool(enformer_hidden_dim)
        self.prediction_head = nn.Linear(enformer_hidden_dim, 1)
        self.mse_loss = nn.MSELoss()
        self.poisson_loss = nn.PoissonNLLLoss(log_input=False)

        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins

        self.human_metrics = nn.ModuleDict(
            {
                "r2_score": R2Score(num_outputs=5313),
            }
        )
        self.mouse_metrics = nn.ModuleDict(
            {
                "r2_score": R2Score(num_outputs=1643),
            }
        )

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
    ):
        """
        X (tensor): (sample * haplotype, length, 4) or (sample * haplotype, length) or (sample, length, 4) or (sample, haplotype, length, 4) or (sample, haplotype, length)
        """
        if not return_base_predictions:
            if X.shape[-1] != 4:
                X = seq_indices_to_one_hot(
                    X
                )  # (S * H, L, 4) or (S, H, L, 4) or (S, L, 4)
            if len(X.shape) == 4:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            X = self.base(
                X,
                return_only_embeddings=True,
                target_length=self.hparams.n_total_bins,
            )  # (S * H, n_total_bins, enformer_hidden_dim)

            assert X.shape[1] == self.hparams.n_total_bins
            X = X[:, self.center_start : self.center_end, :]
            X = self.attention_pool(X)  # (S * H, enformer_hidden_dim)
            Y = self.prediction_head(X)  # (S * H, 1)
            Y = rearrange(Y, "(S H) 1 -> S H", H=2)
            Y = Y.mean(dim=1)
            return Y
        else:
            Y = self.base(X, head=base_predictions_head, target_length=896)

        return Y

    def training_step(self, batch, batch_idx):
        total_loss = 0.0

        for i, dl_batch in enumerate(batch):
            loss = 0.0

            if i == 0:  # this is the pairwise data
                X1, X2, Y = (
                    dl_batch["seq1"],
                    dl_batch["seq2"],
                    dl_batch["z_diff"].float(),
                )
                X = torch.cat([X1, X2], dim=0)
                if X.shape[-1] != 4:
                    X = rearrange(X, "S H L -> (S H) L")
                    X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
                else:
                    X = rearrange(X, "S H L NC -> (S H) L NC")  # (S * H, L, 4)
                Y_hat = self(X)
                Y_hat = Y_hat[: X1.shape[0]] - Y_hat[X1.shape[0] :]
                mse_loss = self.mse_loss(Y_hat, Y)
                self.log("train/pairwise_mse_loss", mse_loss)
                loss += mse_loss
            elif i == 1:  # this is the original human training data
                X, Y = dl_batch["seq"], dl_batch["y"]
                Y_hat = self(
                    X, return_base_predictions=True, base_predictions_head="human"
                )
                poisson_loss = self.poisson_loss(Y_hat, Y)
                self.log("train/human_poisson_loss", poisson_loss)
                loss += poisson_loss
            elif i == 2:  # this is the original mouse training data
                X, Y = dl_batch["seq"], dl_batch["y"]
                Y_hat = self(
                    X, return_base_predictions=True, base_predictions_head="mouse"
                )
                poisson_loss = self.poisson_loss(Y_hat, Y)
                self.log("train/mouse_poisson_loss", poisson_loss)
                loss += poisson_loss
            else:
                raise ValueError(f"Invalid number of dataloaders: {i+1}")

            total_loss += loss

        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log("train/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:  # this is the pairwise data
            X1, X2, Y = (
                batch["seq1"],
                batch["seq2"],
                batch["z_diff"].float(),
            )
            X = torch.cat([X1, X2], dim=0)
            if X.shape[-1] != 4:
                X = rearrange(X, "S H L -> (S H) L")
                X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
            else:
                X = rearrange(X, "S H L NC -> (S H) L NC")  # (S * H, L, 4)
            Y_hat = self(X)
            Y_hat = Y_hat[: X1.shape[0]] - Y_hat[X1.shape[0] :]
            mse_loss = self.mse_loss(Y_hat, Y)
            self.log("val/pairwise_mse_loss", mse_loss, sync_dist=True, on_epoch=True)

        elif dataloader_idx == 1:  # this is the original human training data
            X, Y = batch["seq"], batch["y"]
            Y_hat = self(X, return_base_predictions=True, base_predictions_head="human")
            poisson_loss = self.poisson_loss(Y_hat, Y)
            self.log(
                "val/human_poisson_loss", poisson_loss, sync_dist=True, on_epoch=True
            )
            Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
            Y = Y.reshape(-1, Y.shape[-1])
            self.human_metrics["r2_score"](Y_hat, Y)
            self.log(
                "val/human_r2_score",
                self.human_metrics["r2_score"],
                sync_dist=True,
                on_epoch=True,
            )

        elif dataloader_idx == 2:  # this is the original mouse training data
            X, Y = batch["seq"], batch["y"]
            Y_hat = self(X, return_base_predictions=True, base_predictions_head="mouse")
            poisson_loss = self.poisson_loss(Y_hat, Y)
            self.log(
                "val/mouse_poisson_loss", poisson_loss, sync_dist=True, on_epoch=True
            )
            Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
            Y = Y.reshape(-1, Y.shape[-1])
            self.mouse_metrics["r2_score"](Y_hat, Y)
            self.log(
                "val/mouse_r2_score",
                self.mouse_metrics["r2_score"],
                sync_dist=True,
                on_epoch=True,
            )

        else:
            raise ValueError(f"Invalid number of dataloaders: {dataloader_idx+1}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
        )
        return optimizer


class PairwiseWithOriginalDataJointTrainingAndPairwiseMPRAFloatPrecision(
    L.LightningModule
):
    def __init__(
        self,
        lr: float,
        n_total_bins: int,
        num_cells: int,
        cell_names: list,
        avg_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cell_names = cell_names
        self.num_cells = num_cells

        if checkpoint is None:
            self.base = BaseEnformer.from_pretrained(
                "EleutherAI/enformer-official-rough"
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
                "EleutherAI/enformer-official-rough"
            )
            self.base.load_state_dict(new_state_dict)

        # personalized gene expression
        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins
        enformer_hidden_dim = 2 * self.base.dim
        self.attention_pool = AttentionPool(enformer_hidden_dim)
        self.prediction_head = nn.Linear(enformer_hidden_dim, 1)

        # MPRA
        self.attention_pool_mpra = AttentionPool(enformer_hidden_dim)
        self.prediction_head_mpra = nn.Linear(enformer_hidden_dim, num_cells)

        self.mse_loss = nn.MSELoss()
        self.poisson_loss = nn.PoissonNLLLoss(log_input=False)

        self.human_metrics = nn.ModuleDict(
            {
                "r2_score": R2Score(num_outputs=5313),
            }
        )
        self.mouse_metrics = nn.ModuleDict(
            {
                "r2_score": R2Score(num_outputs=1643),
            }
        )

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        return_mpra_predictions: bool = False,
        base_predictions_head: str = None,
    ):
        """
        X (tensor): (sample * haplotype, length, 4) or (sample * haplotype, length) or (sample, length, 4) or (sample, haplotype, length, 4) or (sample, haplotype, length)
        """
        if not return_base_predictions:
            if not return_base_predictions:
                if X.shape[-1] != 4:
                    X = seq_indices_to_one_hot(
                        X
                    )  # (S * H, L, 4) or (S, H, L, 4) or (S, L, 4)
                if len(X.shape) == 4:
                    X = rearrange(X, "S H L NC -> (S H) L NC")
                X = self.base(
                    X,
                    return_only_embeddings=True,
                    target_length=self.hparams.n_total_bins,
                )  # (S * H, n_total_bins, enformer_hidden_dim)

                assert X.shape[1] == self.hparams.n_total_bins
                X = X[:, self.center_start : self.center_end, :]
                X = self.attention_pool(X)  # (S * H, enformer_hidden_dim)
                Y = self.prediction_head(X)  # (S * H, 1)
                Y = rearrange(Y, "(S H) 1 -> S H", H=2)
                Y = Y.mean(dim=1)
                return Y
            else:
                X = self.base(
                    X,
                    return_only_embeddings=True,
                    target_length=-1,
                )
                X = self.attention_pool_mpra(X)
                Y = self.prediction_head_mpra(X)
                return Y
        else:
            Y = self.base(X, head=base_predictions_head, target_length=896)

        return Y

    def training_step(self, batch, batch_idx):
        total_loss = 0.0

        for i, dl_batch in enumerate(batch):
            loss = 0.0

            if i == 0:  # this is the pairwise data
                X1, X2, Y = (
                    dl_batch["seq1"],
                    dl_batch["seq2"],
                    dl_batch["z_diff"].float(),
                )
                X = torch.cat([X1, X2], dim=0)
                X = rearrange(X, "S H L -> (S H) L")
                X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
                Y_hat = self(X)
                Y_hat = Y_hat[: X1.shape[0]] - Y_hat[X1.shape[0] :]
                mse_loss = self.mse_loss(Y_hat, Y)
                self.log("train/pairwise_mse_loss", mse_loss)
                loss += mse_loss
            elif i == 1:  # this is the original human training data
                X, Y = dl_batch["seq"], dl_batch["y"]
                Y_hat = self(
                    X, return_base_predictions=True, base_predictions_head="human"
                )
                poisson_loss = self.poisson_loss(Y_hat, Y)
                self.log("train/human_poisson_loss", poisson_loss)
                loss += poisson_loss
            elif i == 2:  # this is the original mouse training data
                X, Y = dl_batch["seq"], dl_batch["y"]
                Y_hat = self(
                    X, return_base_predictions=True, base_predictions_head="mouse"
                )
                poisson_loss = self.poisson_loss(Y_hat, Y)
                self.log("train/mouse_poisson_loss", poisson_loss)
                loss += poisson_loss
            elif i == 3:  # this is the MPRA data
                ref_seq, alt_seq, variant_effect, mask = (
                    dl_batch["ref_seq"],
                    dl_batch["alt_seq"],
                    dl_batch["variant_effect"].float(),
                    dl_batch["mask"],
                )
                ref_seq_pred = self(ref_seq, return_mpra_predictions=True)
                alt_seq_pred = self(alt_seq, return_mpra_predictions=True)
                variant_effect_pred = alt_seq_pred - ref_seq_pred
                mse_loss = self.mse_loss(
                    variant_effect_pred[mask], variant_effect[mask]
                )
                self.log("train/MPRA_mse_loss", mse_loss)
                loss += mse_loss
                # also log per cell type mse loss
                for i, cell_name in enumerate(self.cell_names):
                    mask_cell = mask[:, i]
                    if mask_cell.sum() == 0:
                        continue
                    mse_loss_cell = self.mse_loss(
                        variant_effect_pred[mask_cell, i], variant_effect[mask_cell, i]
                    )
                    self.log(f"train/MPRA_mse_loss_{cell_name}", mse_loss_cell)
            else:
                raise ValueError(f"Invalid number of dataloaders: {i+1}")

            total_loss += loss

        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log("train/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:  # this is the pairwise data
            X1, X2, Y = (
                batch["seq1"],
                batch["seq2"],
                batch["z_diff"].float(),
            )
            X = torch.cat([X1, X2], dim=0)
            X = rearrange(X, "S H L -> (S H) L")
            X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
            Y_hat = self(X)
            Y_hat = Y_hat[: X1.shape[0]] - Y_hat[X1.shape[0] :]
            mse_loss = self.mse_loss(Y_hat, Y)
            self.log("val/pairwise_mse_loss", mse_loss, sync_dist=True, on_epoch=True)

        elif dataloader_idx == 1:  # this is the original human training data
            X, Y = batch["seq"], batch["y"]
            Y_hat = self(X, return_base_predictions=True, base_predictions_head="human")
            poisson_loss = self.poisson_loss(Y_hat, Y)
            self.log(
                "val/human_poisson_loss", poisson_loss, sync_dist=True, on_epoch=True
            )
            Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
            Y = Y.reshape(-1, Y.shape[-1])
            self.human_metrics["r2_score"](Y_hat, Y)
            self.log(
                "val/human_r2_score",
                self.human_metrics["r2_score"],
                sync_dist=True,
                on_epoch=True,
            )

        elif dataloader_idx == 2:  # this is the original mouse training data
            X, Y = batch["seq"], batch["y"]
            Y_hat = self(X, return_base_predictions=True, base_predictions_head="mouse")
            poisson_loss = self.poisson_loss(Y_hat, Y)
            self.log(
                "val/mouse_poisson_loss", poisson_loss, sync_dist=True, on_epoch=True
            )
            Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
            Y = Y.reshape(-1, Y.shape[-1])
            self.mouse_metrics["r2_score"](Y_hat, Y)
            self.log(
                "val/mouse_r2_score",
                self.mouse_metrics["r2_score"],
                sync_dist=True,
                on_epoch=True,
            )

        elif dataloader_idx == 3:  # this is the MPRA data
            ref_seq, alt_seq, variant_effect, mask = (
                batch["ref_seq"],
                batch["alt_seq"],
                batch["variant_effect"].float(),
                batch["mask"],
            )
            ref_seq_pred = self(ref_seq, return_mpra_predictions=True)
            alt_seq_pred = self(alt_seq, return_mpra_predictions=True)
            variant_effect_pred = alt_seq_pred - ref_seq_pred
            mse_loss = self.mse_loss(variant_effect_pred[mask], variant_effect[mask])

            self.log("val/MPRA_mse_loss", mse_loss, sync_dist=True, on_epoch=True)
            # also log per cell type mse loss
            for i, cell_name in enumerate(self.cell_names):
                mask_cell = mask[:, i]
                if mask_cell.sum() == 0:
                    continue
                mse_loss_cell = self.mse_loss(
                    variant_effect_pred[mask_cell, i], variant_effect[mask_cell, i]
                )
                self.log(
                    f"val/MPRA_mse_loss_{cell_name}",
                    mse_loss_cell,
                    sync_dist=True,
                    on_epoch=True,
                )

        else:
            raise ValueError(f"Invalid number of dataloaders: {dataloader_idx+1}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
        )
        return optimizer


class PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision(
    L.LightningModule
):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        use_scheduler: bool,
        warmup_steps: int,
        n_total_bins: int,
        pairwise_output_head_upweight_factor: float = 1.0,  # this is so that the pairwise output is upweighted to have the same influence on the loss as the rest of the outputs
        sum_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
        pairwise_output_head_name="human",
        pairwise_output_head_ind=5110,  # this is the CAGE GM12878 cell line output head
    ):
        super().__init__()
        self.save_hyperparameters()
        if checkpoint is None:
            self.base = BaseEnformer.from_pretrained(
                "EleutherAI/enformer-official-rough"
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
                "EleutherAI/enformer-official-rough"
            )
            self.base.load_state_dict(new_state_dict)

        self.pairwise_output_head_name = pairwise_output_head_name
        self.pairwise_output_head_ind = pairwise_output_head_ind
        self.center_start = (n_total_bins - sum_center_n_bins) // 2
        self.center_end = self.center_start + sum_center_n_bins

        self.classification_loss = nn.BCELoss()
        self.poisson_loss = nn.PoissonNLLLoss(log_input=False)

        self.all_metrics = torchmetrics.MetricCollection(
            {
                "pairwise_classification_accuracy": torchmetrics.Accuracy("binary"),
                "pairwise_classification_precision": torchmetrics.Precision("binary"),
                "pairwise_classification_recall": torchmetrics.Recall("binary"),
                "pairwise_classification_auroc": torchmetrics.AUROC("binary"),
                "human_r2_score": R2Score(num_outputs=5313),
                "mouse_r2_score": R2Score(num_outputs=1643),
                "pairwise_output_head_r2_score": R2Score(num_outputs=1),
            }
        )

        self.train_metrics = self.all_metrics.clone(prefix="train/")
        self.val_metrics = self.all_metrics.clone(prefix="val/")

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
    ):
        """
        X (tensor): (sample * haplotype, length, 4) or (sample * haplotype, length) or (sample, length, 4) or (sample, haplotype, length, 4) or (sample, haplotype, length)
        """
        if not return_base_predictions:
            if X.shape[-1] != 4:
                X = seq_indices_to_one_hot(
                    X
                )  # (S * H, L, 4) or (S, H, L, 4) or (S, L, 4)
            if len(X.shape) == 4:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            X = self.base(
                X,
                head=self.pairwise_output_head_name,
                target_length=self.hparams.n_total_bins,
            )  # (S * H, n_total_bins, num_outputs)
            assert X.shape[1] == self.hparams.n_total_bins

            X = X[
                :, :, self.pairwise_output_head_ind
            ]  # get the selected output, (S * H, n_total_bins)
            X = X[
                :, self.center_start : self.center_end
            ]  # get the center bins, (S * H, sum_center_n_bins)
            X = X.sum(dim=1)  # sum the center bins, (S * H)
            Y = rearrange(X, "(S H) -> S H", H=2)  # (S, H)
            Y = Y.mean(dim=1)  # (S)
            return Y
        else:
            Y = self.base(X, head=base_predictions_head, target_length=896)

        return Y

    def compute_skellum_prob_after_anscombe_transform(self, Y1, Y2):
        """
        Here, Y1 and Y2 are assumed to follow the Poisson distribution.
        This function computes the probability of Y1 > Y2 after the Anscombe transform.
        Y1 (tensor): (sample, )
        Y2 (tensor): (sample, )
        """
        Y1 = 2 * torch.sqrt(Y1 + 3 / 8)
        Y2 = 2 * torch.sqrt(Y2 + 3 / 8)

        normal_approx_mean = Y1 - Y2
        normal_approx_std = np.sqrt(2.0)
        skellum_prob = 1.0 - torch.special.ndtr(-normal_approx_mean / normal_approx_std)
        return skellum_prob

    def training_step(self, batch, batch_idx):
        total_loss = 0.0

        for i, dl_batch in enumerate(batch):
            loss = 0.0

            if i == 0:  # this is the pairwise data
                X1, X2, Y = (
                    dl_batch["seq1"],
                    dl_batch["seq2"],
                    dl_batch["Y"].float(),
                )
                X = torch.cat([X1, X2], dim=0)
                if X.shape[-1] != 4:
                    X = rearrange(X, "S H L -> (S H) L")
                    X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
                else:
                    X = rearrange(X, "S H L NC -> (S H) L NC")  # (S * H, L, 4)
                Y_hat = self(X)
                Y1_hat = Y_hat[: X1.shape[0]]
                Y2_hat = Y_hat[X1.shape[0] :]
                skellum_prob = self.compute_skellum_prob_after_anscombe_transform(
                    Y1_hat, Y2_hat
                )
                loss += self.classification_loss(skellum_prob, Y)
                self.log("train/pairwise_classification_loss", loss)

                self.train_metrics["pairwise_classification_accuracy"](skellum_prob, Y)
                self.train_metrics["pairwise_classification_precision"](skellum_prob, Y)
                self.train_metrics["pairwise_classification_recall"](skellum_prob, Y)
                self.train_metrics["pairwise_classification_auroc"](skellum_prob, Y)
                self.log(
                    "train/pairwise_classification_accuracy",
                    self.train_metrics["pairwise_classification_accuracy"],
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    "train/pairwise_classification_precision",
                    self.train_metrics["pairwise_classification_precision"],
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    "train/pairwise_classification_recall",
                    self.train_metrics["pairwise_classification_recall"],
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    "train/pairwise_classification_auroc",
                    self.train_metrics["pairwise_classification_auroc"],
                    on_step=False,
                    on_epoch=True,
                )

            elif i == 1:  # this is the original human training data
                X, Y = dl_batch["seq"], dl_batch["y"]
                Y_hat = self(
                    X, return_base_predictions=True, base_predictions_head="human"
                )
                poisson_loss = self.poisson_loss(Y_hat, Y)
                self.log("train/human_poisson_loss", poisson_loss)
                loss += poisson_loss

                self.train_metrics["human_r2_score"](
                    Y_hat.reshape(-1, Y_hat.shape[-1]), Y.reshape(-1, Y.shape[-1])
                )
                self.log(
                    "train/human_r2_score",
                    self.train_metrics["human_r2_score"],
                    on_step=False,
                    on_epoch=True,
                )

            elif i == 2:  # this is the original mouse training data
                X, Y = dl_batch["seq"], dl_batch["y"]
                Y_hat = self(
                    X, return_base_predictions=True, base_predictions_head="mouse"
                )
                poisson_loss = self.poisson_loss(Y_hat, Y)
                self.log("train/mouse_poisson_loss", poisson_loss)
                loss += poisson_loss

                self.train_metrics["mouse_r2_score"](
                    Y_hat.reshape(-1, Y_hat.shape[-1]), Y.reshape(-1, Y.shape[-1])
                )
                self.log(
                    "train/mouse_r2_score",
                    self.train_metrics["mouse_r2_score"],
                    on_step=False,
                    on_epoch=True,
                )

            else:
                raise ValueError(f"Invalid number of dataloaders: {i+1}")

            # upweight the pairwise output head
            if (i == 1 and self.pairwise_output_head_name == "human") or (
                i == 2 and self.pairwise_output_head_name == "mouse"
            ):
                Y_hat_pairwise_output_head = Y_hat[:, :, self.pairwise_output_head_ind]
                Y_pairwise_output_head = Y[:, :, self.pairwise_output_head_ind]
                pairwise_output_head_poission_loss = self.poisson_loss(
                    Y_hat_pairwise_output_head, Y_pairwise_output_head
                )
                self.log(
                    "train/pairwise_output_head_poisson_loss",
                    pairwise_output_head_poission_loss,
                )
                upweighted_pairwise_output_head_poission_loss = (
                    pairwise_output_head_poission_loss
                    * self.hparams.pairwise_output_head_upweight_factor
                )
                self.log(
                    "train/upweighted_pairwise_output_head_poisson_loss",
                    upweighted_pairwise_output_head_poission_loss,
                )
                loss += upweighted_pairwise_output_head_poission_loss

                Y_hat_pairwise_output_head = Y_hat_pairwise_output_head.reshape(-1)
                Y_pairwise_output_head = Y_pairwise_output_head.reshape(-1)
                self.train_metrics["pairwise_output_head_r2_score"](
                    Y_hat_pairwise_output_head, Y_pairwise_output_head
                )
                self.log(
                    "train/pairwise_output_head_r2_score",
                    self.train_metrics["pairwise_output_head_r2_score"],
                    on_step=False,
                    on_epoch=True,
                )

            total_loss += loss

        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        if self.hparams.weight_decay is not None:
            self.log(
                "train/weight_decay",
                self.trainer.optimizers[0].param_groups[0]["weight_decay"],
            )
        self.log("train/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:  # this is the pairwise data
            X1, X2, Y = (
                batch["seq1"],
                batch["seq2"],
                batch["Y"].float(),
            )
            X = torch.cat([X1, X2], dim=0)
            if X.shape[-1] != 4:
                X = rearrange(X, "S H L -> (S H) L")
                X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
            else:
                X = rearrange(X, "S H L NC -> (S H) L NC")  # (S * H, L, 4)
            Y_hat = self(X)
            Y1_hat = Y_hat[: X1.shape[0]]
            Y2_hat = Y_hat[X1.shape[0] :]
            skellum_prob = self.compute_skellum_prob_after_anscombe_transform(
                Y1_hat, Y2_hat
            )
            loss = self.classification_loss(skellum_prob, Y)
            self.log(
                "val/pairwise_classification_loss", loss, sync_dist=True, on_epoch=True
            )

            self.val_metrics["pairwise_classification_accuracy"](skellum_prob, Y)
            self.val_metrics["pairwise_classification_precision"](skellum_prob, Y)
            self.val_metrics["pairwise_classification_recall"](skellum_prob, Y)
            self.val_metrics["pairwise_classification_auroc"](skellum_prob, Y)
            self.log(
                "val/pairwise_classification_accuracy",
                self.val_metrics["pairwise_classification_accuracy"],
                sync_dist=True,
                on_epoch=True,
            )
            self.log(
                "val/pairwise_classification_precision",
                self.val_metrics["pairwise_classification_precision"],
                sync_dist=True,
                on_epoch=True,
            )
            self.log(
                "val/pairwise_classification_recall",
                self.val_metrics["pairwise_classification_recall"],
                sync_dist=True,
                on_epoch=True,
            )
            self.log(
                "val/pairwise_classification_auroc",
                self.val_metrics["pairwise_classification_auroc"],
                sync_dist=True,
                on_epoch=True,
            )

        elif dataloader_idx == 1:  # this is the original human training data
            X, Y = batch["seq"], batch["y"]
            Y_hat = self(X, return_base_predictions=True, base_predictions_head="human")
            poisson_loss = self.poisson_loss(Y_hat, Y)
            self.log(
                "val/human_poisson_loss", poisson_loss, sync_dist=True, on_epoch=True
            )

            self.val_metrics["human_r2_score"](
                Y_hat.reshape(-1, Y_hat.shape[-1]), Y.reshape(-1, Y.shape[-1])
            )
            self.log(
                "val/human_r2_score",
                self.val_metrics["human_r2_score"],
                sync_dist=True,
                on_epoch=True,
            )

        elif dataloader_idx == 2:  # this is the original mouse training data
            X, Y = batch["seq"], batch["y"]
            Y_hat = self(X, return_base_predictions=True, base_predictions_head="mouse")
            poisson_loss = self.poisson_loss(Y_hat, Y)
            self.log(
                "val/mouse_poisson_loss", poisson_loss, sync_dist=True, on_epoch=True
            )

            self.val_metrics["mouse_r2_score"](
                Y_hat.reshape(-1, Y_hat.shape[-1]), Y.reshape(-1, Y.shape[-1])
            )
            self.log(
                "val/mouse_r2_score",
                self.val_metrics["mouse_r2_score"],
                sync_dist=True,
                on_epoch=True,
            )

        else:
            raise ValueError(f"Invalid number of dataloaders: {dataloader_idx+1}")

        if (dataloader_idx == 1 and self.pairwise_output_head_name == "human") or (
            dataloader_idx == 2 and self.pairwise_output_head_name == "mouse"
        ):
            Y_hat_pairwise_output_head = Y_hat[:, :, self.pairwise_output_head_ind]
            Y_pairwise_output_head = Y[:, :, self.pairwise_output_head_ind]
            pairwise_output_head_poission_loss = self.poisson_loss(
                Y_hat_pairwise_output_head, Y_pairwise_output_head
            )
            self.log(
                "val/pairwise_output_head_poisson_loss",
                pairwise_output_head_poission_loss,
                sync_dist=True,
                on_epoch=True,
            )

            Y_hat_pairwise_output_head = Y_hat_pairwise_output_head.reshape(-1)
            Y_pairwise_output_head = Y_pairwise_output_head.reshape(-1)
            self.val_metrics["pairwise_output_head_r2_score"](
                Y_hat_pairwise_output_head, Y_pairwise_output_head
            )
            self.log(
                "val/pairwise_output_head_r2_score",
                self.val_metrics["pairwise_output_head_r2_score"],
                sync_dist=True,
                on_epoch=True,
            )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            if "seq1" in batch and "seq2" in batch:  # this is the pairwise data
                X1, X2 = batch["seq1"], batch["seq2"]
                X = torch.cat([X1, X2], dim=0)
                if X.shape[-1] != 4:
                    X = rearrange(X, "S H L -> (S H) L")
                    X = seq_indices_to_one_hot(X)
                else:
                    X = rearrange(X, "S H L NC -> (S H) L NC")
                Y_hat = self(X)
                Y1_hat = Y_hat[: X1.shape[0]]
                Y2_hat = Y_hat[X1.shape[0] :]
                skellum_prob = self.compute_skellum_prob_after_anscombe_transform(
                    Y1_hat, Y2_hat
                )
                if "Y" in batch:
                    Y = batch["Y"].float()
                    return {
                        "Y1_hat": Y1_hat,
                        "Y2_hat": Y2_hat,
                        "skellum_prob": skellum_prob,
                        "Y": Y,
                    }
                else:
                    return {
                        "Y1_hat": Y1_hat,
                        "Y2_hat": Y2_hat,
                        "skellum_prob": skellum_prob,
                    }
            elif "seq" in batch:  # this is the individual sample data
                X = batch["seq"]
                Y_hat = self(X)
                if "y" in batch:
                    Y = batch["y"].float()
                    return {"Y_hat": Y_hat, "Y": Y}
                else:
                    return {"Y_hat": Y_hat}
            else:
                raise ValueError("Invalid batch")

        elif dataloader_idx == 1:  # this is the original human training data
            X = batch["seq"]
            Y_hat = self(X, return_base_predictions=True, base_predictions_head="human")
            if "Y" in batch:
                Y = batch["Y"]
                return {"Y_hat": Y_hat, "Y": Y}
            else:
                return Y_hat

        elif dataloader_idx == 2:  # this is the original mouse training data
            X = batch["seq"]
            Y_hat = self(X, return_base_predictions=True, base_predictions_head="mouse")
            if "Y" in batch:
                Y = batch["Y"]
                return {"Y_hat": Y_hat, "Y": Y}
            else:
                return Y_hat

    def configure_optimizers(self):
        if self.hparams.weight_decay is None:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.hparams.lr,
            )
        else:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

        if self.hparams.use_scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.warmup_steps,
                max_epochs=self.trainer.max_steps,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler_config]

        return optimizer


class PairwiseClassificationFloatPrecision(L.LightningModule):
    def __init__(
        self,
        lr: float,
        n_total_bins: int,
        sum_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
        pairwise_output_head_name="human",
        pairwise_output_head_ind=5110,  # this is the CAGE GM12878 cell line output head
    ):
        super().__init__()
        self.save_hyperparameters()
        if checkpoint is None:
            self.base = BaseEnformer.from_pretrained(
                "EleutherAI/enformer-official-rough"
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
                "EleutherAI/enformer-official-rough"
            )
            self.base.load_state_dict(new_state_dict)

        self.pairwise_output_head_name = pairwise_output_head_name
        self.pairwise_output_head_ind = pairwise_output_head_ind
        self.center_start = (n_total_bins - sum_center_n_bins) // 2
        self.center_end = self.center_start + sum_center_n_bins

        self.classification_loss = nn.BCELoss()

        self.pairwise_classification_metrics = nn.ModuleDict(
            {
                "Accuracy": Accuracy("binary"),
                "Precision": Precision("binary"),
                "Recall": Recall("binary"),
                "AUROC": AUROC("binary"),
            }
        )

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
    ):
        """
        X (tensor): (sample * haplotype, length, 4) or (sample * haplotype, length) or (sample, length, 4) or (sample, haplotype, length, 4) or (sample, haplotype, length)
        """
        if not return_base_predictions:
            if X.shape[-1] != 4:
                X = seq_indices_to_one_hot(
                    X
                )  # (S * H, L, 4) or (S, H, L, 4) or (S, L, 4)
            if len(X.shape) == 4:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            X = self.base(
                X,
                head=self.pairwise_output_head_name,
                target_length=self.hparams.n_total_bins,
            )  # (S * H, n_total_bins, num_outputs)
            assert X.shape[1] == self.hparams.n_total_bins

            X = X[
                :, :, self.pairwise_output_head_ind
            ]  # get the selected output, (S * H, n_total_bins)
            X = X[
                :, self.center_start : self.center_end
            ]  # get the center bins, (S * H, sum_center_n_bins)
            X = X.sum(dim=1)  # sum the center bins, (S * H)
            Y = rearrange(X, "(S H) -> S H", H=2)  # (S, H)
            Y = Y.mean(dim=1)  # (S)
            return Y
        else:
            Y = self.base(X, head=base_predictions_head, target_length=896)

        return Y

    def compute_skellum_prob_after_anscombe_transform(self, Y1, Y2):
        """
        Here, Y1 and Y2 are assumed to follow the Poisson distribution.
        This function computes the probability of Y1 > Y2 after the Anscombe transform.
        Y1 (tensor): (sample, )
        Y2 (tensor): (sample, )
        """
        Y1 = 2 * torch.sqrt(Y1 + 3 / 8)
        Y2 = 2 * torch.sqrt(Y2 + 3 / 8)

        normal_approx_mean = Y1 - Y2
        normal_approx_std = np.sqrt(2.0)
        skellum_prob = 1.0 - torch.special.ndtr(-normal_approx_mean / normal_approx_std)
        return skellum_prob

    def training_step(self, batch, batch_idx):
        X1, X2, Y = (
            batch["seq1"],
            batch["seq2"],
            batch["Y"].float(),
        )
        X = torch.cat([X1, X2], dim=0)
        if X.shape[-1] != 4:
            X = rearrange(X, "S H L -> (S H) L")
            X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
        else:
            X = rearrange(X, "S H L NC -> (S H) L NC")  # (S * H, L, 4)
        Y_hat = self(X)
        Y1_hat = Y_hat[: X1.shape[0]]
        Y2_hat = Y_hat[X1.shape[0] :]
        skellum_prob = self.compute_skellum_prob_after_anscombe_transform(
            Y1_hat, Y2_hat
        )
        total_loss = self.classification_loss(skellum_prob, Y)
        self.log("train/pairwise_classification_loss", total_loss)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        X1, X2, Y = (
            batch["seq1"],
            batch["seq2"],
            batch["Y"].float(),
        )
        X = torch.cat([X1, X2], dim=0)
        if X.shape[-1] != 4:
            X = rearrange(X, "S H L -> (S H) L")
            X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
        else:
            X = rearrange(X, "S H L NC -> (S H) L NC")  # (S * H, L, 4)
        Y_hat = self(X)
        Y1_hat = Y_hat[: X1.shape[0]]
        Y2_hat = Y_hat[X1.shape[0] :]
        skellum_prob = self.compute_skellum_prob_after_anscombe_transform(
            Y1_hat, Y2_hat
        )
        loss = self.classification_loss(skellum_prob, Y)
        self.log(
            "val/pairwise_classification_loss", loss, sync_dist=True, on_epoch=True
        )
        self.pairwise_classification_metrics["Accuracy"](skellum_prob, Y.long())
        self.pairwise_classification_metrics["Precision"](skellum_prob, Y.long())
        self.pairwise_classification_metrics["Recall"](skellum_prob, Y.long())
        self.pairwise_classification_metrics["AUROC"](skellum_prob, Y.long())
        self.log(
            "val/pairwise_accuracy",
            self.pairwise_classification_metrics["Accuracy"],
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "val/pairwise_precision",
            self.pairwise_classification_metrics["Precision"],
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "val/pairwise_recall",
            self.pairwise_classification_metrics["Recall"],
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "val/pairwise_auroc",
            self.pairwise_classification_metrics["AUROC"],
            sync_dist=True,
            on_epoch=True,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if "seq1" in batch and "seq2" in batch:  # this is the pairwise data
            X1, X2 = batch["seq1"], batch["seq2"]
            X = torch.cat([X1, X2], dim=0)
            if X.shape[-1] != 4:
                X = rearrange(X, "S H L -> (S H) L")
                X = seq_indices_to_one_hot(X)
            else:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            Y_hat = self(X)
            Y1_hat = Y_hat[: X1.shape[0]]
            Y2_hat = Y_hat[X1.shape[0] :]
            skellum_prob = self.compute_skellum_prob_after_anscombe_transform(
                Y1_hat, Y2_hat
            )
            if "Y" in batch:
                Y = batch["Y"].float()
                return {
                    "Y1_hat": Y1_hat,
                    "Y2_hat": Y2_hat,
                    "skellum_prob": skellum_prob,
                    "Y": Y,
                }
            else:
                return {
                    "Y1_hat": Y1_hat,
                    "Y2_hat": Y2_hat,
                    "skellum_prob": skellum_prob,
                }
        elif "seq" in batch:  # this is the individual sample data
            X = batch["seq"]
            Y_hat = self(X)
            if "y" in batch:
                Y = batch["y"].float()
                return {"Y_hat": Y_hat, "Y": Y}
            else:
                return {"Y_hat": Y_hat}
        else:
            raise ValueError("Invalid batch")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
        )
        return optimizer
