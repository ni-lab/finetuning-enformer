import os
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from datasets import *
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from models import *

torch.manual_seed(97)
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("train_data_path", type=str)
    parser.add_argument("val_data_path", type=str)
    parser.add_argument("test_data_path", type=str)
    parser.add_argument("run_name", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--enformer_checkpoint", type=str, default=None)
    parser.add_argument("--state_dict_subset_prefix", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    train_ds = SampleNormalizedWithGeneDataset(args.train_data_path)
    val_ds = SampleNormalizedWithGeneDataset(args.val_data_path)
    test_ds = SampleNormalizedWithGeneDataset(args.test_data_path)
    all_genes = sorted(
        list(
            set(
                list(train_ds.unique_genes)
                + list(val_ds.unique_genes)
                + list(test_ds.unique_genes)
            )
        )
    )
    print("Total number of genes: ", len(all_genes))

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
    early_stopping_cb = EarlyStopping(
        monitor="val/mse_loss", mode="min", patience=args.patience
    )
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
    )

    model = SingleWithGeneFinetuned(
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_total_bins=train_ds.get_total_n_bins(),
        all_genes=all_genes,
        checkpoint=args.enformer_checkpoint,
        state_dict_subset_prefix=args.state_dict_subset_prefix,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
