import os
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from datasets import PairwiseDataset
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from models import PairwiseFinetuned
from utils import BaseEnformerFreezeUnfreeze

torch.manual_seed(97)
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("train_data_path", type=str)
    parser.add_argument("val_data_path", type=str)
    parser.add_argument("run_name", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--unfreeze_at_epoch", type=int, default=1)
    parser.add_argument("--initial_denom_lr", type=float, default=1.0)
    parser.add_argument("--train_bn", action=BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_n_pairs", type=int, default=100_000)
    parser.add_argument("--val_n_pairs", type=int, default=5_000)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--enformer_checkpoint", type=str, default=None)
    parser.add_argument("--state_dict_subset_prefix", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    train_ds = PairwiseDataset(args.train_data_path, n_pairs=args.train_n_pairs)
    val_ds = PairwiseDataset(args.val_data_path, n_pairs=args.val_n_pairs)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=32
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=32
    )

    logger = WandbLogger(
        project="enformer-finetune", name=args.run_name, save_dir=args.save_dir
    )
    checkpointing_cb = ModelCheckpoint(
        monitor="val/mse_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping_cb = EarlyStopping(monitor="val/mse_loss", mode="min", patience=2)
    # finetuning_cb = BaseEnformerFreezeUnfreeze(
    #     unfreeze_at_epoch=args.unfreeze_at_epoch,
    #     initial_denom_lr=args.initial_denom_lr,
    #     train_bn=args.train_bn,
    # )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        gradient_clip_val=0.2,
        logger=logger,
        default_root_dir=args.save_dir,
        callbacks=[checkpointing_cb, early_stopping_cb],
        # callbacks=[early_stopping_cb, finetuning_cb],
    )

    model = PairwiseFinetuned(
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_total_bins=train_ds.get_total_n_bins(),
        checkpoint=args.enformer_checkpoint,
        state_dict_subset_prefix=args.state_dict_subset_prefix,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
