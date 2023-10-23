import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from enformer_pytorch import Enformer as BaseEnformer


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
    def __init__(self, lr: float, n_total_bins: int, avg_center_n_bins: int = 10):
        super().__init__()
        self.save_hyperparameters()
        self.base = BaseEnformer.from_pretrained(
            "EleutherAI/enformer-official-rough", target_length=n_total_bins
        )
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
        return mse_loss

    def validation_step(self, batch, batch_idx):
        X1, X2, Y = batch["seq1"], batch["seq2"], batch["z_diff"].float()
        mse_loss = self.get_mse_loss(X1, X2, Y)
        self.log("val/mse_loss", mse_loss)
        return mse_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr
        )
