import pdb

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from enformer_pytorch import Enformer as BaseEnformer
from enformer_pytorch.data import seq_indices_to_one_hot, str_to_one_hot
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef


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


class PairwiseWithOriginalDataJointTraining(L.LightningModule):
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
        self.poisson_loss = nn.PoissonNLLLoss(log_input=False, eps=1e-4)

        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins

        self.human_metrics = nn.ModuleDict(
            {
                #                 "spearman_corr": SpearmanCorrCoef(num_outputs=5313),
                "pearson_corr": PearsonCorrCoef(num_outputs=5313),
                "mse": MeanSquaredError(),
            }
        )
        self.mouse_metrics = nn.ModuleDict(
            {
                #                 "spearman_corr": SpearmanCorrCoef(num_outputs=1643),
                "pearson_corr": PearsonCorrCoef(num_outputs=1643),
                "mse": MeanSquaredError(),
            }
        )

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
    ):
        """
        X (tensor): (sample * haplotype, length, 4) or (sample, length, 4)
        """
        if not return_base_predictions:
            X = self.base(
                X, return_only_embeddings=True, target_length=self.hparams.n_total_bins
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
#                     dl_batch["z_diff"].float(),
                    dl_batch["z_diff"].half(),
                )
                X = torch.cat([X1, X2], dim=0)
                X = rearrange(X, "S H L -> (S H) L")
                X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
                X = X.half()
                Y_hat = self(X)
                Y_hat = Y_hat[: X1.shape[0]] - Y_hat[X1.shape[0] :]
                mse_loss = self.mse_loss(Y_hat, Y)
                self.log("train/pairwise_mse_loss", mse_loss)
                loss += mse_loss
            elif i == 1:  # this is the original human training data
#                 X, Y = dl_batch["seq"], dl_batch["y"]
                X, Y = dl_batch["seq"].half(), dl_batch["y"].half()
                Y_hat = self(
                    X, return_base_predictions=True, base_predictions_head="human"
                )
                poisson_loss = self.poisson_loss(Y_hat, Y)
                self.log("train/human_poisson_loss", poisson_loss)
                loss += poisson_loss
            elif i == 2:  # this is the original mouse training data
#                 X, Y = dl_batch["seq"], dl_batch["y"]
                X, Y = dl_batch["seq"].half(), dl_batch["y"].half()
                Y_hat = self(
                    X, return_base_predictions=True, base_predictions_head="mouse"
                )
                poisson_loss = self.poisson_loss(Y_hat, Y)
                self.log("train/mouse_poisson_loss", poisson_loss)
                loss += poisson_loss
            else:
                raise ValueError(f"Invalid number of dataloaders: {i+1}")

            total_loss += loss

        self.log("train/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:  # this is the pairwise data
            X1, X2, Y = (
                batch["seq1"],
                batch["seq2"],
#                 batch["z_diff"].float(),
                batch["z_diff"].half(),
            )
            X = torch.cat([X1, X2], dim=0)
            X = rearrange(X, "S H L -> (S H) L")
            X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
            X = X.half()
            Y_hat = self(X)
            Y_hat = Y_hat[: X1.shape[0]] - Y_hat[X1.shape[0] :]
            mse_loss = self.mse_loss(Y_hat, Y)
            self.log("val/pairwise_mse_loss", mse_loss)

        elif dataloader_idx == 1:  # this is the original human training data
#             X, Y = batch["seq"], batch["y"]
            X, Y = batch["seq"].half(), batch["y"].half()
            Y_hat = self(X, return_base_predictions=True, base_predictions_head="human")
            poisson_loss = self.poisson_loss(Y_hat, Y)
            self.log("val/human_poisson_loss", poisson_loss)
            Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
            Y = Y.reshape(-1, Y.shape[-1])
            #             self.human_metrics["spearman_corr"].update(Y_hat, Y)
            self.human_metrics["pearson_corr"].update(Y_hat, Y)
            self.human_metrics["mse"].update(Y_hat, Y)

        elif dataloader_idx == 2:  # this is the original mouse training data
#             X, Y = batch["seq"], batch["y"]
            X, Y = batch["seq"].half(), batch["y"].half()
            Y_hat = self(X, return_base_predictions=True, base_predictions_head="mouse")
            poisson_loss = self.poisson_loss(Y_hat, Y)
            self.log("val/mouse_poisson_loss", poisson_loss)
            Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
            Y = Y.reshape(-1, Y.shape[-1])
            #             self.mouse_metrics["spearman_corr"].update(Y_hat, Y)
            self.mouse_metrics["pearson_corr"].update(Y_hat, Y)
            self.mouse_metrics["mse"].update(Y_hat, Y)

        else:
            raise ValueError(f"Invalid number of dataloaders: {dataloader_idx+1}")

    def on_validation_epoch_end(self):
        for metric_name, metric in self.human_metrics.items():
            self.log(
                f"val/human_{metric_name}", metric.compute().mean(), sync_dist=True
            )
            metric.reset()

        for metric_name, metric in self.mouse_metrics.items():
            self.log(
                f"val/mouse_{metric_name}", metric.compute().mean(), sync_dist=True
            )
            metric.reset()

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
