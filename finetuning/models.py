import pdb

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from einops import rearrange
from enformer_pytorch import Enformer as BaseEnformer
from enformer_pytorch.config_enformer import EnformerConfig
from enformer_pytorch.data import seq_indices_to_one_hot, str_to_one_hot
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
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


class BaseModule(L.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        use_scheduler: bool,
        warmup_steps: int,
        checkpoint=None,
        state_dict_subset_prefix=None,
        use_random_init=False,
        add_gaussian_noise_to_pretrained_weights=False,
        gaussian_noise_std_multiplier=1,
        freeze_cnn=False,
        freeze_transformer=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.warmup_steps = warmup_steps

        if use_random_init:
            assert (
                add_gaussian_noise_to_pretrained_weights is False
            ), "Cannot use random init and add Gaussian noise to pretrained weights at the same time"
            print("Using random init")
            self.base = BaseEnformer(EnformerConfig())
        else:
            if checkpoint is None:
                self.base = BaseEnformer.from_pretrained(
                    "EleutherAI/enformer-official-rough"
                )
            else:
                checkpoint = torch.load(checkpoint)
                new_state_dict = {}
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

            # to determine the usefulness of pretraining, we can perturb the pretrained weights with Gaussian noise and see if the model still performs well after fine-tuning
            # for each weight matrix, we add Gaussian noise with standard deviation equal to original_std * gaussian_noise_std_multiplier
            # do not add noise to the batch norm parameters
            if add_gaussian_noise_to_pretrained_weights:
                print(
                    "Injecting Gaussian noise to pretrained weights with std multiplier: ",
                    gaussian_noise_std_multiplier,
                )

                batch_norm_parameter_names = [
                    "stem.1.fn.0.weight",
                    "stem.1.fn.0.bias",
                    "conv_tower.0.0.0.weight",
                    "conv_tower.0.0.0.bias",
                    "conv_tower.0.1.fn.0.weight",
                    "conv_tower.0.1.fn.0.bias",
                    "conv_tower.1.0.0.weight",
                    "conv_tower.1.0.0.bias",
                    "conv_tower.1.1.fn.0.weight",
                    "conv_tower.1.1.fn.0.bias",
                    "conv_tower.2.0.0.weight",
                    "conv_tower.2.0.0.bias",
                    "conv_tower.2.1.fn.0.weight",
                    "conv_tower.2.1.fn.0.bias",
                    "conv_tower.3.0.0.weight",
                    "conv_tower.3.0.0.bias",
                    "conv_tower.3.1.fn.0.weight",
                    "conv_tower.3.1.fn.0.bias",
                    "conv_tower.4.0.0.weight",
                    "conv_tower.4.0.0.bias",
                    "conv_tower.4.1.fn.0.weight",
                    "conv_tower.4.1.fn.0.bias",
                    "conv_tower.5.0.0.weight",
                    "conv_tower.5.0.0.bias",
                    "conv_tower.5.1.fn.0.weight",
                    "conv_tower.5.1.fn.0.bias",
                    "final_pointwise.1.0.weight",
                    "final_pointwise.1.0.bias",
                ]

                ori_parameter_stds = {}
                bn_parameters_encountered = 0
                for name, param in self.base.named_parameters():
                    if name in batch_norm_parameter_names:
                        print("Skipping ", name)
                        bn_parameters_encountered += 1
                        continue
                    if param.requires_grad:
                        original_std = torch.std(param)
                        param.data += torch.normal(
                            mean=0.0,
                            std=torch.abs(param) * gaussian_noise_std_multiplier,
                        )
                        new_std = torch.std(param)
                        print(
                            "Added noise to ",
                            name,
                            " with original std: ",
                            original_std,
                            " and new std: ",
                            new_std,
                        )
                        ori_parameter_stds[name] = original_std.cpu().detach().numpy()
                assert bn_parameters_encountered == len(
                    batch_norm_parameter_names
                ), "Not all batch norm parameters were encountered"

                # make sure that the noise is added correctly
                for name, param in self.base.named_parameters():
                    if name in batch_norm_parameter_names:
                        continue
                    if param.requires_grad:
                        assert (
                            torch.std(param).cpu().detach().numpy()
                            != ori_parameter_stds[name]
                        )

        if freeze_cnn:
            initial_number_of_trainable_parameters = sum(
                p.numel() for p in self.base.parameters() if p.requires_grad
            )
            for name, param in self.base.named_parameters():
                # names start with stem or conv_tower
                if name.startswith("stem") or name.startswith("conv_tower"):
                    param.requires_grad = False
            final_number_of_trainable_parameters = sum(
                p.numel() for p in self.base.parameters() if p.requires_grad
            )
            print(
                "CNN frozen, number of trainable parameters reduced from ",
                initial_number_of_trainable_parameters,
                " to ",
                final_number_of_trainable_parameters,
            )
            assert (
                final_number_of_trainable_parameters
                < initial_number_of_trainable_parameters
            )

        if freeze_transformer:
            initial_number_of_trainable_parameters = sum(
                p.numel() for p in self.base.parameters() if p.requires_grad
            )
            for name, param in self.base.named_parameters():
                # names start with transformer
                if name.startswith("transformer"):
                    param.requires_grad = False
            final_number_of_trainable_parameters = sum(
                p.numel() for p in self.base.parameters() if p.requires_grad
            )
            print(
                "Transformer frozen, number of trainable parameters reduced from ",
                initial_number_of_trainable_parameters,
                " to ",
                final_number_of_trainable_parameters,
            )
            assert (
                final_number_of_trainable_parameters
                < initial_number_of_trainable_parameters
            )

    def freeze_all_weights_except_enformer_output_heads(self):
        initial_number_of_trainable_parameters = sum(
            p.numel() for p in self.base.parameters() if p.requires_grad
        )
        for name, param in self.base.named_parameters():
            # names start with _head
            if not name.startswith("_head"):
                param.requires_grad = False
        final_number_of_trainable_parameters = sum(
            p.numel() for p in self.base.parameters() if p.requires_grad
        )
        print(
            "All weights frozen except enformer output heads, number of trainable parameters reduced from ",
            initial_number_of_trainable_parameters,
            " to ",
            final_number_of_trainable_parameters,
        )
        assert (
            final_number_of_trainable_parameters
            < initial_number_of_trainable_parameters
        )

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


class PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision(BaseModule):
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
        use_random_init=False,
        add_gaussian_noise_to_pretrained_weights=False,
        gaussian_noise_std_multiplier=1,
        freeze_cnn=False,
        freeze_transformer=False,
        pairwise_output_head_name="human",
        pairwise_output_head_ind=5110,  # this is the CAGE GM12878 cell line output head
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            use_scheduler=use_scheduler,
            warmup_steps=warmup_steps,
            checkpoint=checkpoint,
            state_dict_subset_prefix=state_dict_subset_prefix,
            use_random_init=use_random_init,
            add_gaussian_noise_to_pretrained_weights=add_gaussian_noise_to_pretrained_weights,
            gaussian_noise_std_multiplier=gaussian_noise_std_multiplier,
            freeze_cnn=freeze_cnn,
            freeze_transformer=freeze_transformer,
        )

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
        no_haplotype: bool = False,
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
            if not no_haplotype:
                Y = rearrange(X, "(S H) -> S H", H=2)  # (S, H)
                Y = Y.mean(dim=1)  # (S)
            else:
                Y = X
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

            elif (
                "seq" in batch and "y" in batch and len(batch["y"].shape) == 3
            ):  # this is the original enformer training data
                X = batch["seq"]
                Y = batch["y"].float()
                base_predictions_head = "human" if Y.shape[2] == 5313 else "mouse"
                Y_hat = self(
                    X,
                    return_base_predictions=True,
                    base_predictions_head=base_predictions_head,
                )
                return {"Y_hat": Y_hat, "Y": Y}

            elif "seq" in batch:  # this is the individual sample data
                X = batch["seq"]
                true_idx = batch["true_idx"]
                if (
                    "is_ref" in batch
                ):  # this is the ISM data, so run forward pass without haplotype averaging
                    Y_hat = self(X, no_haplotype=True)
                    assert Y_hat.shape[0] == X.shape[0]
                else:
                    Y_hat = self(X)
                if "y" in batch:
                    Y = batch["y"].float()
                    return {"Y_hat": Y_hat, "Y": Y, "true_idx": true_idx}
                else:
                    result = {}
                    result["Y_hat"] = Y_hat
                    for key in batch:
                        if key != "seq":
                            result[key] = batch[key]
                    return result
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


class PairwiseClassificationFloatPrecision(BaseModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        use_scheduler: bool,
        warmup_steps: int,
        n_total_bins: int,
        sum_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
        use_random_init=False,
        add_gaussian_noise_to_pretrained_weights=False,
        gaussian_noise_std_multiplier=1,
        freeze_cnn=False,
        freeze_transformer=False,
        pairwise_output_head_name="human",
        pairwise_output_head_ind=5110,  # this is the CAGE GM12878 cell line output head
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            use_scheduler=use_scheduler,
            warmup_steps=warmup_steps,
            checkpoint=checkpoint,
            state_dict_subset_prefix=state_dict_subset_prefix,
            use_random_init=use_random_init,
            add_gaussian_noise_to_pretrained_weights=add_gaussian_noise_to_pretrained_weights,
            gaussian_noise_std_multiplier=gaussian_noise_std_multiplier,
            freeze_cnn=freeze_cnn,
            freeze_transformer=freeze_transformer,
        )

        self.pairwise_output_head_name = pairwise_output_head_name
        self.pairwise_output_head_ind = pairwise_output_head_ind
        self.center_start = (n_total_bins - sum_center_n_bins) // 2
        self.center_end = self.center_start + sum_center_n_bins

        self.classification_loss = nn.BCELoss()

        self.all_metrics = torchmetrics.MetricCollection(
            {
                "pairwise_classification_accuracy": torchmetrics.Accuracy("binary"),
                "pairwise_classification_precision": torchmetrics.Precision("binary"),
                "pairwise_classification_recall": torchmetrics.Recall("binary"),
                "pairwise_classification_auroc": torchmetrics.AUROC("binary"),
            }
        )

        self.train_metrics = self.all_metrics.clone(prefix="train/")
        self.val_metrics = self.all_metrics.clone(prefix="val/")

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
        no_haplotype: bool = False,
    ):
        """
        X (tensor): (sample * haplotype, length, 4) or (sample * haplotype, length) or (sample, length, 4) or (sample, haplotype, length, 4) or (sample, haplotype, length)
        """
        if not return_base_predictions:
            if X.shape[-1] != 4:
                X = seq_indices_to_one_hot(X)
            if len(X.shape) == 4:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            X = self.base(
                X,
                head=self.pairwise_output_head_name,
                target_length=self.hparams.n_total_bins,
            )
            assert X.shape[1] == self.hparams.n_total_bins

            X = X[:, :, self.pairwise_output_head_ind]
            X = X[:, self.center_start : self.center_end]
            X = X.sum(dim=1)
            if not no_haplotype:
                Y = rearrange(X, "(S H) -> S H", H=2)  # (S, H)
                Y = Y.mean(dim=1)  # (S)
            else:
                Y = X
            return Y
        else:
            Y = self.base(X, head=base_predictions_head, target_length=896)

        return Y

    def compute_skellum_prob_after_anscombe_transform(self, Y1, Y2):
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
        loss = self.classification_loss(skellum_prob, Y)
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

        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        if self.hparams.weight_decay is not None:
            self.log(
                "train/weight_decay",
                self.trainer.optimizers[0].param_groups[0]["weight_decay"],
            )

        return loss

    def validation_step(self, batch, batch_idx):
        X1, X2, Y = (
            batch["seq1"],
            batch["seq2"],
            batch["Y"].float(),
        )
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
        loss = self.classification_loss(skellum_prob, Y)
        self.log("val/pairwise_classification_loss", loss)

        self.val_metrics["pairwise_classification_accuracy"](skellum_prob, Y)
        self.val_metrics["pairwise_classification_precision"](skellum_prob, Y)
        self.val_metrics["pairwise_classification_recall"](skellum_prob, Y)
        self.val_metrics["pairwise_classification_auroc"](skellum_prob, Y)
        self.log(
            "val/pairwise_classification_accuracy",
            self.val_metrics["pairwise_classification_accuracy"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/pairwise_classification_precision",
            self.val_metrics["pairwise_classification_precision"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/pairwise_classification_recall",
            self.val_metrics["pairwise_classification_recall"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/pairwise_classification_auroc",
            self.val_metrics["pairwise_classification_auroc"],
            on_step=False,
            on_epoch=True,
        )

    def predict_step(self, batch, batch_idx):
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

        elif (
            "seq" in batch and "y" in batch and len(batch["y"].shape) == 3
        ):  # this is the original enformer training data
            X = batch["seq"]
            Y = batch["y"].float()
            base_predictions_head = "human" if Y.shape[2] == 5313 else "mouse"
            Y_hat = self(
                X,
                return_base_predictions=True,
                base_predictions_head=base_predictions_head,
            )
            return {"Y_hat": Y_hat, "Y": Y}

        elif "seq" in batch:  # this is the individual sample data
            X = batch["seq"]
            true_idx = batch["true_idx"]
            if (
                "is_ref" in batch
            ):  # this is the ISM data, so run forward pass without haplotype averaging
                Y_hat = self(X, no_haplotype=True)
                assert Y_hat.shape[0] == X.shape[0]
            else:
                Y_hat = self(X)
            if "Y" in batch:
                Y = batch["Y"].float()
                return {"Y_hat": Y_hat, "Y": Y, "true_idx": true_idx}
            else:
                result = {}
                result["Y_hat"] = Y_hat
                for key in batch:
                    if key != "seq":
                        result[key] = batch[key]
                return result


class PairwiseRegressionFloatPrecision(BaseModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        use_scheduler: bool,
        warmup_steps: int,
        n_total_bins: int,
        avg_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
        use_random_init=False,
        add_gaussian_noise_to_pretrained_weights=False,
        gaussian_noise_std_multiplier=1,
        freeze_cnn=False,
        freeze_transformer=False,
        finetune_enformer_output_heads_only=False,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            use_scheduler=use_scheduler,
            warmup_steps=warmup_steps,
            checkpoint=checkpoint,
            state_dict_subset_prefix=state_dict_subset_prefix,
            use_random_init=use_random_init,
            add_gaussian_noise_to_pretrained_weights=add_gaussian_noise_to_pretrained_weights,
            gaussian_noise_std_multiplier=gaussian_noise_std_multiplier,
            freeze_cnn=freeze_cnn,
            freeze_transformer=freeze_transformer,
        )

        enformer_hidden_dim = 2 * self.base.dim
        self.attention_pool = AttentionPool(enformer_hidden_dim)
        self.prediction_head = nn.Linear(enformer_hidden_dim, 1)
        self.mse_loss = nn.MSELoss()

        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins

        self.finetune_enformer_output_heads_only = finetune_enformer_output_heads_only
        self.poisson_loss = nn.PoissonNLLLoss(log_input=False)
        if self.finetune_enformer_output_heads_only:
            self.all_metrics = torchmetrics.MetricCollection(
                {
                    "human_r2_score": R2Score(num_outputs=5313),
                    "mouse_r2_score": R2Score(num_outputs=1643),
                }
            )

            self.train_metrics = self.all_metrics.clone(prefix="train/")
            self.val_metrics = self.all_metrics.clone(prefix="val/")

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
        no_haplotype: bool = False,
    ):
        """
        X (tensor): (sample * haplotype, length, 4) or (sample * haplotype, length) or (sample, length, 4) or (sample, haplotype, length, 4) or (sample, haplotype, length)
        """
        if not return_base_predictions:
            if X.shape[-1] != 4:
                X = seq_indices_to_one_hot(X)
            if len(X.shape) == 4:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            X = self.base(
                X,
                return_only_embeddings=True,
                target_length=self.hparams.n_total_bins,
            )
            assert X.shape[1] == self.hparams.n_total_bins
            X = X[:, self.center_start : self.center_end, :]
            X = self.attention_pool(X)
            Y = self.prediction_head(X)
            if not no_haplotype:
                Y = rearrange(Y, "(S H) 1 -> S H", H=2)
                Y = Y.mean(dim=1)
            else:
                Y = Y.squeeze()
            return Y
        else:
            Y = self.base(X, head=base_predictions_head, target_length=896)

        return Y

    def training_step(self, batch, batch_idx):
        if not self.finetune_enformer_output_heads_only:
            X1, X2, Y = (
                batch["seq1"],
                batch["seq2"],
                batch["z_diff"].float(),
            )
            X = torch.cat([X1, X2], dim=0)
            if X.shape[-1] != 4:
                X = rearrange(X, "S H L -> (S H) L")
                X = seq_indices_to_one_hot(X)
            else:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            Y_hat = self(X)
            Y1_hat = Y_hat[: X1.shape[0]]
            Y2_hat = Y_hat[X1.shape[0] :]
            Y_diff = Y1_hat - Y2_hat
            loss = self.mse_loss(Y_diff, Y)
            self.log(
                "train/pairwise_regression_loss", loss, on_step=True, on_epoch=True
            )

            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
            if self.hparams.weight_decay is not None:
                self.log(
                    "train/weight_decay",
                    self.trainer.optimizers[0].param_groups[0]["weight_decay"],
                )

            return loss
        else:
            total_loss = 0.0

            for i, dl_batch in enumerate(batch):
                loss = 0.0

                if i == 0:  # this is the original human training data
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

                elif i == 1:  # this is the original mouse training data
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
        if not self.finetune_enformer_output_heads_only:
            X1, X2, Y = (
                batch["seq1"],
                batch["seq2"],
                batch["z_diff"].float(),
            )
            X = torch.cat([X1, X2], dim=0)
            if X.shape[-1] != 4:
                X = rearrange(X, "S H L -> (S H) L")
                X = seq_indices_to_one_hot(X)
            else:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            Y_hat = self(X)
            Y1_hat = Y_hat[: X1.shape[0]]
            Y2_hat = Y_hat[X1.shape[0] :]
            Y_diff = Y1_hat - Y2_hat
            loss = self.mse_loss(Y_diff, Y)
            self.log(
                "val/pairwise_regression_loss", loss, sync_dist=True, on_epoch=True
            )

        else:
            if dataloader_idx == 0:  # this is the original human training data
                X, Y = batch["seq"], batch["y"]
                Y_hat = self(
                    X, return_base_predictions=True, base_predictions_head="human"
                )
                poisson_loss = self.poisson_loss(Y_hat, Y)
                self.log(
                    "val/human_poisson_loss",
                    poisson_loss,
                    sync_dist=True,
                    on_epoch=True,
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

            elif dataloader_idx == 1:  # this is the original mouse training data
                X, Y = batch["seq"], batch["y"]
                Y_hat = self(
                    X, return_base_predictions=True, base_predictions_head="mouse"
                )
                poisson_loss = self.poisson_loss(Y_hat, Y)
                self.log(
                    "val/mouse_poisson_loss",
                    poisson_loss,
                    sync_dist=True,
                    on_epoch=True,
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

    def predict_step(self, batch, batch_idx):
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
            Y_diff = Y1_hat - Y2_hat
            if "z_diff" in batch:
                Y = batch["z_diff"].float()
                return {
                    "Y1_hat": Y1_hat,
                    "Y2_hat": Y2_hat,
                    "Y_diff": Y_diff,
                    "Y": Y,
                }
            else:
                return {
                    "Y1_hat": Y1_hat,
                    "Y2_hat": Y2_hat,
                    "Y_diff": Y_diff,
                }

        elif (
            "seq" in batch and "y" in batch and len(batch["y"].shape) == 3
        ):  # this is the original enformer training data
            X = batch["seq"]
            Y = batch["y"].float()
            base_predictions_head = "human" if Y.shape[2] == 5313 else "mouse"
            Y_hat = self(
                X,
                return_base_predictions=True,
                base_predictions_head=base_predictions_head,
            )
            return {"Y_hat": Y_hat, "Y": Y}

        elif "seq" in batch:  # this is the individual sample data
            X = batch["seq"]
            true_idx = batch["true_idx"]
            if (
                "is_ref" in batch
            ):  # this is the ISM data, so run forward pass without haplotype averaging
                Y_hat = self(X, no_haplotype=True)
                assert Y_hat.shape[0] == X.shape[0]
            else:
                Y_hat = self(X)
            if "z" in batch:
                Y = batch["z"].float()
                return {"Y_hat": Y_hat, "Y": Y, "true_idx": true_idx}
            else:
                result = {}
                result["Y_hat"] = Y_hat
                for key in batch:
                    if key != "seq":
                        result[key] = batch[key]
                return result


class PairwiseRegressionWithOriginalDataJointTrainingFloatPrecision(BaseModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        use_scheduler: bool,
        warmup_steps: int,
        n_total_bins: int,
        avg_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
        use_random_init=False,
        add_gaussian_noise_to_pretrained_weights=False,
        gaussian_noise_std_multiplier=1,
        freeze_cnn=False,
        freeze_transformer=False,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            use_scheduler=use_scheduler,
            warmup_steps=warmup_steps,
            checkpoint=checkpoint,
            state_dict_subset_prefix=state_dict_subset_prefix,
            use_random_init=use_random_init,
            add_gaussian_noise_to_pretrained_weights=add_gaussian_noise_to_pretrained_weights,
            gaussian_noise_std_multiplier=gaussian_noise_std_multiplier,
            freeze_cnn=freeze_cnn,
            freeze_transformer=freeze_transformer,
        )

        enformer_hidden_dim = 2 * self.base.dim
        self.attention_pool = AttentionPool(enformer_hidden_dim)
        self.prediction_head = nn.Linear(enformer_hidden_dim, 1)
        self.mse_loss = nn.MSELoss()
        self.poisson_loss = nn.PoissonNLLLoss(log_input=False)

        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins

        self.all_metrics = torchmetrics.MetricCollection(
            {
                "human_r2_score": R2Score(num_outputs=5313),
                "mouse_r2_score": R2Score(num_outputs=1643),
            }
        )

        self.train_metrics = self.all_metrics.clone(prefix="train/")
        self.val_metrics = self.all_metrics.clone(prefix="val/")

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
        no_haplotype: bool = False,
    ):
        """
        X (tensor): (sample * haplotype, length, 4) or (sample * haplotype, length) or (sample, length, 4) or (sample, haplotype, length, 4) or (sample, haplotype, length)
        """
        if not return_base_predictions:
            if X.shape[-1] != 4:
                X = seq_indices_to_one_hot(X)
            if len(X.shape) == 4:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            X = self.base(
                X,
                return_only_embeddings=True,
                target_length=self.hparams.n_total_bins,
            )
            assert X.shape[1] == self.hparams.n_total_bins
            X = X[:, self.center_start : self.center_end, :]
            X = self.attention_pool(X)
            Y = self.prediction_head(X)
            if not no_haplotype:
                Y = rearrange(Y, "(S H) 1 -> S H", H=2)
                Y = Y.mean(dim=1)
            else:
                Y = Y.squeeze()
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
                    X = rearrange(X, "S H L NC -> (S H) L NC")
                Y_hat = self(X)
                Y1_hat = Y_hat[: X1.shape[0]]
                Y2_hat = Y_hat[X1.shape[0] :]
                Y_diff = Y1_hat - Y2_hat
                loss = self.mse_loss(Y_diff, Y)
                self.log(
                    "train/pairwise_regression_loss", loss, on_step=True, on_epoch=True
                )
                total_loss += loss

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
                batch["z_diff"].float(),
            )
            X = torch.cat([X1, X2], dim=0)
            if X.shape[-1] != 4:
                X = rearrange(X, "S H L -> (S H) L")
                X = seq_indices_to_one_hot(X)
            else:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            Y_hat = self(X)
            Y1_hat = Y_hat[: X1.shape[0]]
            Y2_hat = Y_hat[X1.shape[0] :]
            Y_diff = Y1_hat - Y2_hat
            loss = self.mse_loss(Y_diff, Y)
            self.log(
                "val/pairwise_regression_loss", loss, sync_dist=True, on_epoch=True
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
                Y_diff = Y1_hat - Y2_hat
                if "z_diff" in batch:
                    Y = batch["z_diff"].float()
                    return {
                        "Y1_hat": Y1_hat,
                        "Y2_hat": Y2_hat,
                        "Y_diff": Y_diff,
                        "Y": Y,
                    }
                else:
                    return {
                        "Y1_hat": Y1_hat,
                        "Y2_hat": Y2_hat,
                        "Y_diff": Y_diff,
                    }

            elif (
                "seq" in batch and "y" in batch and len(batch["y"].shape) == 3
            ):  # this is the original enformer training data
                X = batch["seq"]
                Y = batch["y"].float()
                base_predictions_head = "human" if Y.shape[2] == 5313 else "mouse"
                Y_hat = self(
                    X,
                    return_base_predictions=True,
                    base_predictions_head=base_predictions_head,
                )
                return {"Y_hat": Y_hat, "Y": Y}

            elif "seq" in batch:  # this is the individual sample data
                X = batch["seq"]
                true_idx = batch["true_idx"]
                if (
                    "is_ref" in batch
                ):  # this is the ISM data, so run forward pass without haplotype averaging
                    Y_hat = self(X, no_haplotype=True)
                    assert Y_hat.shape[0] == X.shape[0]
                else:
                    Y_hat = self(X)
                if "z" in batch:
                    Y = batch["z"].float()
                    return {"Y_hat": Y_hat, "Y": Y, "true_idx": true_idx}
                else:
                    result = {}
                    result["Y_hat"] = Y_hat
                    for key in batch:
                        if key != "seq":
                            result[key] = batch[key]
                    return result

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


class PairwiseRegressionOnCountsWithOriginalDataJointTrainingFloatPrecision(BaseModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        use_scheduler: bool,
        warmup_steps: int,
        n_total_bins: int,
        avg_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
        use_random_init=False,
        add_gaussian_noise_to_pretrained_weights=False,
        gaussian_noise_std_multiplier=1,
        freeze_cnn=False,
        freeze_transformer=False,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            use_scheduler=use_scheduler,
            warmup_steps=warmup_steps,
            checkpoint=checkpoint,
            state_dict_subset_prefix=state_dict_subset_prefix,
            use_random_init=use_random_init,
            add_gaussian_noise_to_pretrained_weights=add_gaussian_noise_to_pretrained_weights,
            gaussian_noise_std_multiplier=gaussian_noise_std_multiplier,
            freeze_cnn=freeze_cnn,
            freeze_transformer=freeze_transformer,
        )

        enformer_hidden_dim = 2 * self.base.dim
        self.attention_pool = AttentionPool(enformer_hidden_dim)
        self.prediction_head = nn.Linear(enformer_hidden_dim, 1)
        self.poisson_loss = nn.PoissonNLLLoss(log_input=False)

        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins

        self.all_metrics = torchmetrics.MetricCollection(
            {
                "human_r2_score": R2Score(num_outputs=5313),
                "mouse_r2_score": R2Score(num_outputs=1643),
            }
        )

        self.train_metrics = self.all_metrics.clone(prefix="train/")
        self.val_metrics = self.all_metrics.clone(prefix="val/")

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
        no_haplotype: bool = False,
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
            if not no_haplotype:
                Y = rearrange(Y, "(S H) 1 -> S H", H=2)
                Y = Y.mean(dim=1)
            else:
                Y = Y.squeeze()
            return Y
        else:
            Y = self.base(X, head=base_predictions_head, target_length=896)

        return Y

    def __smape(self, Y1, Y2):
        """
        Y1 (tensor): (sample,)
        Y2 (tensor): (sample,)
        """
        return torch.mean(2 * torch.abs(Y1 - Y2) / (torch.abs(Y1) + torch.abs(Y2)))

    def training_step(self, batch, batch_idx):
        total_loss = 0.0

        for i, dl_batch in enumerate(batch):
            loss = 0.0

            if i == 0:  # this is the pairwise data
                X1, X2, Y1, Y2 = (
                    dl_batch["seq1"],
                    dl_batch["seq2"],
                    dl_batch["Y1"].float(),
                    dl_batch["Y2"].float(),
                )
                X = torch.cat([X1, X2], dim=0)
                if X.shape[-1] != 4:
                    X = rearrange(X, "S H L -> (S H) L")
                    X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
                else:
                    X = rearrange(X, "S H L NC -> (S H) L NC")  # (S * H, L, 4)
                Y_diff = Y1 - Y2

                Y_hat = self(X)
                Y1_hat = Y_hat[: X1.shape[0]]
                Y2_hat = Y_hat[X1.shape[0] :]
                Y_diff_hat = Y1_hat - Y2_hat

                # Compute SMAPE loss on individual samples
                single_smape_loss = 0.5 * (
                    self.__smape(Y1_hat, Y1) + self.__smape(Y2_hat, Y2)
                )
                self.log("train/single_smape_loss", single_smape_loss)
                loss += single_smape_loss

                # Compute SMAPE loss on sample pairs
                pairwise_smape_loss = self.__smape(Y_diff_hat, Y_diff)
                self.log("train/pairwise_smape_loss", pairwise_smape_loss)
                loss += pairwise_smape_loss

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
            X1, X2, Y1, Y2 = (
                batch["seq1"],
                batch["seq2"],
                batch["Y1"].float(),
                batch["Y2"].float(),
            )
            X = torch.cat([X1, X2], dim=0)
            if X.shape[-1] != 4:
                X = rearrange(X, "S H L -> (S H) L")
                X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
            else:
                X = rearrange(X, "S H L NC -> (S H) L NC")  # (S * H, L, 4)
            Y_diff = Y1 - Y2

            Y_hat = self(X)
            Y1_hat = Y_hat[: X1.shape[0]]
            Y2_hat = Y_hat[X1.shape[0] :]
            Y_diff_hat = Y1_hat - Y2_hat

            # Compute SMAPE loss on individual samples
            single_smape_loss = 0.5 * (
                self.__smape(Y1_hat, Y1) + self.__smape(Y2_hat, Y2)
            )
            self.log(
                "val/single_smape_loss",
                single_smape_loss,
                sync_dist=True,
                on_epoch=True,
            )

            # Compute SMAPE loss on sample pairs
            pairwise_smape_loss = self.__smape(Y_diff_hat, Y_diff)
            self.log(
                "val/pairwise_smape_loss",
                pairwise_smape_loss,
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
            Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
            Y = Y.reshape(-1, Y.shape[-1])

            self.val_metrics["human_r2_score"](Y_hat, Y)
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
            Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
            Y = Y.reshape(-1, Y.shape[-1])

            self.val_metrics["mouse_r2_score"](Y_hat, Y)
            self.log(
                "val/mouse_r2_score",
                self.val_metrics["mouse_r2_score"],
                sync_dist=True,
                on_epoch=True,
            )

        else:
            raise ValueError(f"Invalid number of dataloaders: {dataloader_idx+1}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            if "seq1" in batch and "seq2" in batch:  # this is the pairwise data
                X1, X2 = batch["seq1"], batch["seq2"]
                X = torch.cat([X1, X2], dim=0)
                if X.shape[-1] != 4:
                    X = rearrange(X, "S H L -> (S H) L")
                    X = seq_indices_to_one_hot(X)  # (S * H, L, 4)
                else:
                    X = rearrange(X, "S H L NC -> (S H) L NC")  # (S * H, L, 4)

                Y_hat = self(X)
                Y1_hat = Y_hat[: X1.shape[0]]
                Y2_hat = Y_hat[X1.shape[0] :]
                Y_diff_hat = Y1_hat - Y2_hat

                if "Y1" in batch and "Y2" in batch:
                    Y1, Y2 = batch["Y1"].float(), batch["Y2"].float()
                    Y_diff = Y1 - Y2

                    # Compute SMAPE loss on individual samples
                    single_smape_loss = 0.5 * (
                        self.__smape(Y1_hat, Y1) + self.__smape(Y2_hat, Y2)
                    )

                    # Compute SMAPE loss on sample pairs
                    pairwise_smape_loss = self.__smape(Y_diff_hat, Y_diff)

                    return {
                        "Y1_hat": Y1_hat,
                        "Y2_hat": Y2_hat,
                        "Y_diff_hat": Y_diff_hat,
                        "single_smape_loss": single_smape_loss,
                        "pairwise_smape_loss": pairwise_smape_loss,
                        "Y1": Y1,
                        "Y2": Y2,
                    }
                else:
                    return {
                        "Y1_hat": Y1_hat,
                        "Y2_hat": Y2_hat,
                        "Y_diff_hat": Y_diff_hat,
                    }

            elif (
                "seq" in batch and "y" in batch and len(batch["y"].shape) == 3
            ):  # this is the original enformer training data
                X = batch["seq"]
                Y = batch["y"].float()
                base_predictions_head = "human" if Y.shape[2] == 5313 else "mouse"
                Y_hat = self(
                    X,
                    return_base_predictions=True,
                    base_predictions_head=base_predictions_head,
                )
                return {"Y_hat": Y_hat, "Y": Y}

            elif "seq" in batch:  # this is the individual sample data
                X = batch["seq"]
                true_idx = batch["true_idx"]
                if (
                    "is_ref" in batch
                ):  # this is the ISM data, so run forward pass without haplotype averaging
                    Y_hat = self(X, no_haplotype=True)
                    assert Y_hat.shape[0] == X.shape[0]
                else:
                    Y_hat = self(X)
                if "y" in batch:
                    Y = batch["y"].float()
                    return {"Y_hat": Y_hat, "Y": Y, "true_idx": true_idx}
                else:
                    result = {}
                    result["Y_hat"] = Y_hat
                    for key in batch:
                        if key != "seq":
                            result[key] = batch[key]
                    return result

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


class SingleRegressionFloatPrecision(BaseModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        use_scheduler: bool,
        warmup_steps: int,
        n_total_bins: int,
        avg_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
        use_random_init=False,
        add_gaussian_noise_to_pretrained_weights=False,
        gaussian_noise_std_multiplier=1,
        freeze_cnn=False,
        freeze_transformer=False,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            use_scheduler=use_scheduler,
            warmup_steps=warmup_steps,
            checkpoint=checkpoint,
            state_dict_subset_prefix=state_dict_subset_prefix,
            use_random_init=use_random_init,
            add_gaussian_noise_to_pretrained_weights=add_gaussian_noise_to_pretrained_weights,
            gaussian_noise_std_multiplier=gaussian_noise_std_multiplier,
            freeze_cnn=freeze_cnn,
            freeze_transformer=freeze_transformer,
        )

        enformer_hidden_dim = 2 * self.base.dim
        self.attention_pool = AttentionPool(enformer_hidden_dim)
        self.prediction_head = nn.Linear(enformer_hidden_dim, 1)
        self.mse_loss = nn.MSELoss()

        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
        no_haplotype: bool = False,
    ):
        """
        X (tensor): (sample * haplotype, length, 4) or (sample * haplotype, length
        """
        if not return_base_predictions:
            if X.shape[-1] != 4:
                X = seq_indices_to_one_hot(X)
            if len(X.shape) == 4:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            X = self.base(
                X,
                return_only_embeddings=True,
                target_length=self.hparams.n_total_bins,
            )
            assert X.shape[1] == self.hparams.n_total_bins
            X = X[:, self.center_start : self.center_end, :]
            X = self.attention_pool(X)
            Y = self.prediction_head(X)
            if not no_haplotype:
                Y = rearrange(Y, "(S H) 1 -> S H", H=2)
                Y = Y.mean(dim=1)
            else:
                Y = Y.squeeze()
            return Y
        else:
            Y = self.base(X, head=base_predictions_head, target_length=896)

        return Y

    def training_step(self, batch, batch_idx):
        X, Y = batch["seq"], batch["z"]
        Y_hat = self(X)
        loss = self.mse_loss(Y_hat, Y)
        self.log("train/mse_loss", loss)

        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        if self.hparams.weight_decay is not None:
            self.log(
                "train/weight_decay",
                self.trainer.optimizers[0].param_groups[0]["weight_decay"],
            )

        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch["seq"], batch["z"]
        Y_hat = self(X)
        loss = self.mse_loss(Y_hat, Y)
        self.log("val/mse_loss", loss, sync_dist=True, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        if (
            "seq" in batch and "y" in batch and len(batch["y"].shape) == 3
        ):  # this is the original enformer training data
            X = batch["seq"]
            Y = batch["y"].float()
            base_predictions_head = "human" if Y.shape[2] == 5313 else "mouse"
            Y_hat = self(
                X,
                return_base_predictions=True,
                base_predictions_head=base_predictions_head,
            )
            return {"Y_hat": Y_hat, "Y": Y}

        X = batch["seq"]
        true_idx = batch["true_idx"]
        if (
            "is_ref" in batch
        ):  # this is the ISM data, so run forward pass without haplotype averaging
            Y_hat = self(X, no_haplotype=True)
            assert Y_hat.shape[0] == X.shape[0]
        else:
            Y_hat = self(X)
        if "z" in batch:
            Y = batch["z"]
            return {"Y_hat": Y_hat, "Y": Y, "true_idx": true_idx}
        else:
            result = {}
            result["Y_hat"] = Y_hat
            for key in batch:
                if key != "seq":
                    result[key] = batch[key]
            return result


class SingleRegressionOnCountsFloatPrecision(BaseModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        use_scheduler: bool,
        warmup_steps: int,
        n_total_bins: int,
        avg_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
        use_random_init=False,
        add_gaussian_noise_to_pretrained_weights=False,
        gaussian_noise_std_multiplier=1,
        freeze_cnn=False,
        freeze_transformer=False,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            use_scheduler=use_scheduler,
            warmup_steps=warmup_steps,
            checkpoint=checkpoint,
            state_dict_subset_prefix=state_dict_subset_prefix,
            use_random_init=use_random_init,
            add_gaussian_noise_to_pretrained_weights=add_gaussian_noise_to_pretrained_weights,
            gaussian_noise_std_multiplier=gaussian_noise_std_multiplier,
            freeze_cnn=freeze_cnn,
            freeze_transformer=freeze_transformer,
        )

        enformer_hidden_dim = 2 * self.base.dim
        self.attention_pool = AttentionPool(enformer_hidden_dim)
        self.prediction_head = nn.Linear(enformer_hidden_dim, 1)

        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins

        self.all_metrics = torchmetrics.MetricCollection(
            {
                "r2_score": R2Score(num_outputs=1),
            }
        )

        self.train_metrics = self.all_metrics.clone(prefix="train/")
        self.val_metrics = self.all_metrics.clone(prefix="val/")

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
        no_haplotype: bool = False,
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
            if not no_haplotype:
                Y = rearrange(Y, "(S H) 1 -> S H", H=2)
                Y = Y.mean(dim=1)
            else:
                Y = Y.squeeze()
            return Y
        else:
            Y = self.base(X, head=base_predictions_head, target_length=896)

        return Y

    def __smape(self, Y1, Y2):
        """
        Y1 (tensor): (sample,)
        Y2 (tensor): (sample,)
        """
        return torch.mean(2 * torch.abs(Y1 - Y2) / (torch.abs(Y1) + torch.abs(Y2)))

    def training_step(self, batch, batch_idx):
        X, Y, Z = batch["seq"], batch["y"], batch["z"]
        Y_hat = self(X)
        loss = self.__smape(Y_hat, Y)
        self.log("train/smape_loss", loss)

        self.train_metrics["r2_score"](Y_hat, Y)
        self.log("train/r2_score", self.train_metrics["r2_score"], on_epoch=True)

        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        if self.hparams.weight_decay is not None:
            self.log(
                "train/weight_decay",
                self.trainer.optimizers[0].param_groups[0]["weight_decay"],
            )

        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, Z = batch["seq"], batch["y"], batch["z"]
        Y_hat = self(X)
        loss = self.__smape(Y_hat, Y)
        self.log("val/smape_loss", loss, sync_dist=True, on_epoch=True)

        self.val_metrics["r2_score"](Y_hat, Y)
        self.log(
            "val/r2_score", self.val_metrics["r2_score"], sync_dist=True, on_epoch=True
        )

    def predict_step(self, batch, batch_idx):
        if (
            "seq" in batch and "y" in batch and len(batch["y"].shape) == 3
        ):  # this is the original enformer training data
            X = batch["seq"]
            Y = batch["y"].float()
            base_predictions_head = "human" if Y.shape[2] == 5313 else "mouse"
            Y_hat = self(
                X,
                return_base_predictions=True,
                base_predictions_head=base_predictions_head,
            )
            return {"Y_hat": Y_hat, "Y": Y}

        X, Y, Z = batch["seq"], batch["y"], batch["z"]
        true_idx = batch["true_idx"]
        if (
            "is_ref" in batch
        ):  # this is the ISM data, so run forward pass without haplotype averaging
            Y_hat = self(X, no_haplotype=True)
            assert Y_hat.shape[0] == X.shape[0]
        else:
            Y_hat = self(X)
        if "y" in batch:
            Y = batch["y"]
            return {"Y_hat": Y_hat, "Y": Y, "true_idx": true_idx}
        else:
            result = {}
            result["Y_hat"] = Y_hat
            for key in batch:
                if key != "seq":
                    result[key] = batch[key]
            return result


class BaselineEnformer(BaseModule):
    def __init__(
        self,
        n_total_bins: int,
        sum_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
        use_random_init=False,
        add_gaussian_noise_to_pretrained_weights=False,
        gaussian_noise_std_multiplier=1,
        freeze_cnn=False,
        freeze_transformer=False,
        output_head_name="human",
        output_head_ind=5110,  # this is the CAGE GM12878 cell line output head
    ):
        super().__init__(
            # dummy values for training hyperparameters since we are only doing inference using this model
            lr=0.0,
            weight_decay=0.0,
            use_scheduler=False,
            warmup_steps=0,
            checkpoint=checkpoint,
            state_dict_subset_prefix=state_dict_subset_prefix,
            use_random_init=use_random_init,
            add_gaussian_noise_to_pretrained_weights=add_gaussian_noise_to_pretrained_weights,
            gaussian_noise_std_multiplier=gaussian_noise_std_multiplier,
            freeze_cnn=freeze_cnn,
            freeze_transformer=freeze_transformer,
        )

        self.output_head_name = output_head_name
        self.output_head_ind = output_head_ind
        self.center_start = (n_total_bins - sum_center_n_bins) // 2
        self.center_end = self.center_start + sum_center_n_bins

    def forward(
        self,
        X,
        no_haplotype: bool = False,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
    ):
        if return_base_predictions:
            Y = self.base(X, head=base_predictions_head, target_length=896)
            return Y

        """
        X (tensor): (sample * haplotype, length, 4) or (sample * haplotype, length) or (sample, length, 4) or (sample, haplotype, length, 4) or (sample, haplotype, length)
        """
        if X.shape[-1] != 4:
            X = seq_indices_to_one_hot(X)
        if len(X.shape) == 4:
            X = rearrange(X, "S H L NC -> (S H) L NC")
        X = self.base(
            X,
            head=self.output_head_name,
            target_length=self.hparams.n_total_bins,
        )
        assert X.shape[1] == self.hparams.n_total_bins

        X = X[:, :, self.output_head_ind]
        X = X[:, self.center_start : self.center_end]
        X = X.sum(dim=1)
        if not no_haplotype:
            Y = rearrange(X, "(S H) -> S H", H=2)  # (S, H)
            Y = Y.mean(dim=1)  # (S)
        else:
            Y = X
        return Y

    def predict_step(self, batch, batch_idx):
        assert (
            "seq" in batch
        ), "BaselineEnformer is only for inference and requires 'seq' in the batch"

        if (
            "seq" in batch and "y" in batch and len(batch["y"].shape) == 3
        ):  # this is the original enformer training data
            X = batch["seq"]
            Y = batch["y"].float()
            base_predictions_head = "human" if Y.shape[2] == 5313 else "mouse"
            Y_hat = self(
                X,
                return_base_predictions=True,
                base_predictions_head=base_predictions_head,
            )
            return {"Y_hat": Y_hat, "Y": Y}

        X = batch["seq"]
        true_idx = batch["true_idx"]
        if (
            "is_ref" in batch
        ):  # this is the ISM data, so run forward pass without haplotype averaging
            Y_hat = self(X, no_haplotype=True)
            assert Y_hat.shape[0] == X.shape[0]
        else:
            Y_hat = self(X)
        if "Y" in batch:
            Y = batch["Y"].float()
            return {"Y_hat": Y_hat, "Y": Y, "true_idx": true_idx}
        else:
            result = {}
            result["Y_hat"] = Y_hat
            for key in batch:
                if key != "seq":
                    result[key] = batch[key]
            return result


class PairwiseRegressionWithMalinoisMPRAJointTrainingFloatPrecision(BaseModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        use_scheduler: bool,
        warmup_steps: int,
        n_total_bins: int,
        n_total_bins_malinois: int,
        malinois_num_cells: int,
        avg_center_n_bins: int = 10,
        checkpoint=None,
        state_dict_subset_prefix=None,
        use_random_init=False,
        add_gaussian_noise_to_pretrained_weights=False,
        gaussian_noise_std_multiplier=1,
        freeze_cnn=False,
        freeze_transformer=False,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            use_scheduler=use_scheduler,
            warmup_steps=warmup_steps,
            checkpoint=checkpoint,
            state_dict_subset_prefix=state_dict_subset_prefix,
            use_random_init=use_random_init,
            add_gaussian_noise_to_pretrained_weights=add_gaussian_noise_to_pretrained_weights,
            gaussian_noise_std_multiplier=gaussian_noise_std_multiplier,
            freeze_cnn=freeze_cnn,
            freeze_transformer=freeze_transformer,
        )

        # output head for personalized gene expression prediction
        enformer_hidden_dim = 2 * self.base.dim
        self.n_total_bins = n_total_bins
        self.attention_pool = AttentionPool(enformer_hidden_dim)
        self.prediction_head = nn.Linear(enformer_hidden_dim, 1)
        self.mse_loss = nn.MSELoss()

        self.center_start = (n_total_bins - avg_center_n_bins) // 2
        self.center_end = self.center_start + avg_center_n_bins

        # output head for Malinois MPRA data
        self.n_total_bins_malinois = n_total_bins_malinois
        self.malinois_num_cells = malinois_num_cells
        self.malinois_output_head = nn.Linear(
            enformer_hidden_dim * n_total_bins_malinois, malinois_num_cells
        )
        self.malinois_mse_loss = nn.MSELoss()

    def forward(
        self,
        X,
        return_base_predictions: bool = False,
        base_predictions_head: str = None,
        no_haplotype: bool = False,
        use_malinois_output_head: bool = False,
    ):
        """
        X (tensor): (sample * haplotype, length, 4) or (sample * haplotype, length) or (sample, length, 4) or (sample, haplotype, length, 4) or (sample, haplotype, length)
        """
        if use_malinois_output_head:
            if X.shape[-1] != 4:
                X = seq_indices_to_one_hot(X)
            if len(X.shape) == 4:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            X = self.base(
                X,
                return_only_embeddings=True,
                target_length=self.hparams.n_total_bins_malinois,
            )
            assert X.shape[1] == self.hparams.n_total_bins_malinois
            X = X.flatten(start_dim=1)
            Y = self.malinois_output_head(
                X
            )  # (S * H, malinois_num_cells) or (S, malinois_num_cells)
            if not no_haplotype:
                Y = rearrange(Y, "(S H) CE -> S H CE", H=2)
                Y = Y.mean(dim=1)
            return Y
        elif not return_base_predictions:
            if X.shape[-1] != 4:
                X = seq_indices_to_one_hot(X)
            if len(X.shape) == 4:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            X = self.base(
                X,
                return_only_embeddings=True,
                target_length=self.hparams.n_total_bins,
            )
            assert X.shape[1] == self.hparams.n_total_bins
            X = X[:, self.center_start : self.center_end, :]
            X = self.attention_pool(X)
            Y = self.prediction_head(X)
            if not no_haplotype:
                Y = rearrange(Y, "(S H) 1 -> S H", H=2)
                Y = Y.mean(dim=1)
            else:
                Y = Y.squeeze()
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
                    X = rearrange(X, "S H L NC -> (S H) L NC")
                Y_hat = self(X)
                Y1_hat = Y_hat[: X1.shape[0]]
                Y2_hat = Y_hat[X1.shape[0] :]
                Y_diff = Y1_hat - Y2_hat
                loss = self.mse_loss(Y_diff, Y)
                self.log(
                    "train/pairwise_regression_loss", loss, on_step=True, on_epoch=True
                )
                total_loss += loss

            elif i == 1:  # this is the Malinois MPRA data
                X_ref, X_alt, Y, mask = (
                    dl_batch["ref_seq"],
                    dl_batch["alt_seq"],
                    dl_batch["variant_effect"].float(),
                    dl_batch["mask"],
                )
                X = torch.cat([X_ref, X_alt], dim=0)
                if X.shape[-1] != 4:
                    X = seq_indices_to_one_hot(X)  # (S, L, 4)
                Y_hat = self(X, no_haplotype=True, use_malinois_output_head=True)
                Y_ref_hat = Y_hat[: X_ref.shape[0]]
                Y_alt_hat = Y_hat[X_ref.shape[0] :]
                Y_diff = Y_ref_hat - Y_alt_hat
                loss = self.malinois_mse_loss(Y_diff[mask], Y[mask])
                self.log(
                    "train/malinois_pairwise_regression_loss",
                    loss,
                    on_step=True,
                    on_epoch=True,
                )
                total_loss += loss

            else:
                raise ValueError(f"Invalid number of dataloaders: {i+1}")

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
                batch["z_diff"].float(),
            )
            X = torch.cat([X1, X2], dim=0)
            if X.shape[-1] != 4:
                X = rearrange(X, "S H L -> (S H) L")
                X = seq_indices_to_one_hot(X)
            else:
                X = rearrange(X, "S H L NC -> (S H) L NC")
            Y_hat = self(X)
            Y1_hat = Y_hat[: X1.shape[0]]
            Y2_hat = Y_hat[X1.shape[0] :]
            Y_diff = Y1_hat - Y2_hat
            loss = self.mse_loss(Y_diff, Y)
            self.log(
                "val/pairwise_regression_loss", loss, sync_dist=True, on_epoch=True
            )

        elif dataloader_idx == 1:  # this is the Malinois MPRA data
            X_ref, X_alt, Y, mask = (
                batch["ref_seq"],
                batch["alt_seq"],
                batch["variant_effect"].float(),
                batch["mask"],
            )
            X = torch.cat([X_ref, X_alt], dim=0)
            if X.shape[-1] != 4:
                X = seq_indices_to_one_hot(X)
            Y_hat = self(X, no_haplotype=True, use_malinois_output_head=True)
            Y_ref_hat = Y_hat[: X_ref.shape[0]]
            Y_alt_hat = Y_hat[X_ref.shape[0] :]
            Y_diff = Y_ref_hat - Y_alt_hat
            loss = self.malinois_mse_loss(Y_diff[mask], Y[mask])
            self.log(
                "val/malinois_pairwise_regression_loss",
                loss,
                sync_dist=True,
                on_epoch=True,
            )

        else:
            raise ValueError(f"Invalid number of dataloaders: {dataloader_idx+1}")

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
                Y_diff = Y1_hat - Y2_hat
                if "z_diff" in batch:
                    Y = batch["z_diff"].float()
                    return {
                        "Y1_hat": Y1_hat,
                        "Y2_hat": Y2_hat,
                        "Y_diff": Y_diff,
                        "Y": Y,
                    }
                else:
                    return {
                        "Y1_hat": Y1_hat,
                        "Y2_hat": Y2_hat,
                        "Y_diff": Y_diff,
                    }

            elif (
                "seq" in batch and "y" in batch and len(batch["y"].shape) == 3
            ):  # this is the original enformer training data
                X = batch["seq"]
                Y = batch["y"].float()
                base_predictions_head = "human" if Y.shape[2] == 5313 else "mouse"
                Y_hat = self(
                    X,
                    return_base_predictions=True,
                    base_predictions_head=base_predictions_head,
                )
                return {"Y_hat": Y_hat, "Y": Y}

            elif "seq" in batch:  # this is the individual sample data
                X = batch["seq"]
                true_idx = batch["true_idx"]
                if (
                    "is_ref" in batch
                ):  # this is the ISM data, so run forward pass without haplotype averaging
                    Y_hat = self(X, no_haplotype=True)
                    assert Y_hat.shape[0] == X.shape[0]
                else:
                    Y_hat = self(X)
                if "z" in batch:
                    Y = batch["z"].float()
                    return {"Y_hat": Y_hat, "Y": Y, "true_idx": true_idx}
                else:
                    result = {}
                    result["Y_hat"] = Y_hat
                    for key in batch:
                        if key != "seq":
                            result[key] = batch[key]
                    return result

        elif dataloader_idx == 1:  # this is the Malinois MPRA data
            X_ref, X_alt = batch["ref_seq"], batch["alt_seq"]
            X = torch.cat([X_ref, X_alt], dim=0)
            if X.shape[-1] != 4:
                X = seq_indices_to_one_hot(X)
            Y_hat = self(X, no_haplotype=True, use_malinois_output_head=True)
            Y_ref_hat = Y_hat[: X_ref.shape[0]]
            Y_alt_hat = Y_hat[X_ref.shape[0] :]
            Y_diff = Y_ref_hat - Y_alt_hat
            if "variant_effect" in batch:
                Y = batch["variant_effect"].float()
                mask = batch["mask"]
                return {
                    "Y_ref_hat": Y_ref_hat,
                    "Y_alt_hat": Y_alt_hat,
                    "Y_diff": Y_diff,
                    "Y": Y,
                    "mask": mask,
                }
            else:
                return {
                    "Y_ref_hat": Y_ref_hat,
                    "Y_alt_hat": Y_alt_hat,
                    "Y_diff": Y_diff,
                }
