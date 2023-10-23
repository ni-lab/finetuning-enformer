import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from datasets import PairwiseDataset
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from models import PairwiseFinetuned
from utils import BaseEnformerFreezeUnfreeze


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("train_data_path", type=str)
    parser.add_argument("val_data_path", type=str)
    parser.add_argument("run_name", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze_at_epoch", type=int, default=1)
    parser.add_argument("--initial_denom_lr", type=float, default=1.0)
    parser.add_argument("--train_bn", action=BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_n_pairs", type=int, default=100_000)
    parser.add_argument("--val_n_pairs", type=int, default=5_000)
    parser.add_argument("--max_epochs", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    train_ds = PairwiseDataset(args.train_data_path, n_pairs=args.train_n_pairs)
    val_ds = PairwiseDataset(args.val_data_path, n_pairs=args.val_n_pairs)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    logger = WandbLogger(
        project="enformer-finetune", name=args.run_name, save_dir=args.save_dir
    )
    early_stopping_cb = EarlyStopping(monitor="val/mse_loss", mode="min", patience=2)
    finetuning_cb = BaseEnformerFreezeUnfreeze(
        unfreeze_at_epoch=args.unfreeze_at_epoch,
        initial_denom_lr=args.initial_denom_lr,
        train_bn=args.train_bn,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        max_epochs=args.max_epochs,
        gradient_clip_val=0.2,
        logger=logger,
        default_root_dir=args.save_dir,
        callbacks=[early_stopping_cb, finetuning_cb],
    )

    model = PairwiseFinetuned(lr=args.lr, n_total_bins=train_ds.get_total_n_bins())
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
