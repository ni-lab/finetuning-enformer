import os
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import torch
from datasets import (EnformerDataset, PairwiseRegressionOnCountsH5Dataset,
                      PairwiseRegressionOnCountsH5DatasetDynamicSampling)
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from models import \
    PairwiseRegressionOnCountsWithOriginalDataJointTrainingFloatPrecision

np.random.seed(97)
torch.manual_seed(97)
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("train_data_path", type=str)
    parser.add_argument("val_data_path", type=str)
    parser.add_argument("enformer_data_path", type=str)
    parser.add_argument("run_name", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--use_scheduler", action=BooleanOptionalAction, default=False)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_n_pairs_per_gene", type=int, default=250)
    parser.add_argument("--val_n_pairs_per_gene", type=int, default=100)
    parser.add_argument("--seqlen", type=int, default=128 * 384)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--enformer_checkpoint", type=str, default=None)
    parser.add_argument("--state_dict_subset_prefix", type=str, default=None)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument(
        "--resume_from_checkpoint", action=BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pairwise_train_ds = PairwiseRegressionOnCountsH5DatasetDynamicSampling(
        args.train_data_path,
        n_pairs_per_gene=args.train_n_pairs_per_gene,
        seqlen=args.seqlen,
        random_seed=args.data_seed,
    )
    pairwise_val_ds = PairwiseRegressionOnCountsH5Dataset(
        args.val_data_path,
        n_pairs_per_gene=args.val_n_pairs_per_gene,
        seqlen=args.seqlen,
    )

    human_enformer_train_ds = EnformerDataset(
        args.enformer_data_path,
        species="human",
        split="train",
        reverse_complement=True,
        random_shift=True,
    )
    human_enformer_val_ds = EnformerDataset(
        args.enformer_data_path,
        species="human",
        split="val",
        reverse_complement=False,
        random_shift=False,
    )

    mouse_enformer_train_ds = EnformerDataset(
        args.enformer_data_path,
        species="mouse",
        split="train",
        reverse_complement=True,
        random_shift=True,
    )
    mouse_enformer_val_ds = EnformerDataset(
        args.enformer_data_path,
        species="mouse",
        split="val",
        reverse_complement=False,
        random_shift=False,
    )

    train_dl = CombinedLoader(
        [
            torch.utils.data.DataLoader(
                pairwise_train_ds, batch_size=args.batch_size, shuffle=True
            ),
            torch.utils.data.DataLoader(
                human_enformer_train_ds, batch_size=args.batch_size
            ),
            torch.utils.data.DataLoader(
                mouse_enformer_train_ds, batch_size=args.batch_size
            ),
        ],
        mode="max_size_cycle",
    )

    val_dl = [
        torch.utils.data.DataLoader(
            pairwise_val_ds, batch_size=args.batch_size, shuffle=False
        ),
        torch.utils.data.DataLoader(human_enformer_val_ds, batch_size=args.batch_size),
        torch.utils.data.DataLoader(mouse_enformer_val_ds, batch_size=args.batch_size),
    ]

    run_save_dir = os.path.join(
        args.save_dir, args.run_name + f"_data_seed_{args.data_seed}"
    )
    os.makedirs(run_save_dir, exist_ok=True)

    logs_dir = os.path.join(run_save_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    ckpts_dir = os.path.join(run_save_dir, "checkpoints")
    os.makedirs(ckpts_dir, exist_ok=True)

    logger = WandbLogger(
        project="enformer-finetune",
        name=args.run_name + f"_data_seed_{args.data_seed}",
        save_dir=logs_dir,
    )

    checkpointing_cb = ModelCheckpoint(
        dirpath=ckpts_dir,
        monitor="val/pairwise_smape_loss/dataloader_idx_0",
        mode="min",
        save_top_k=3,
    )

    early_stopping_cb = EarlyStopping(
        monitor="val/pairwise_smape_loss/dataloader_idx_0",
        mode="min",
        patience=5,
    )

    os.environ["SLURM_JOB_NAME"] = "interactive"
    # get number of gpus
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")

    # print hyperparameters
    # we accumulate gradients over 64 samples to match the original Enformer model's batch size
    # commented out line shows the explicit calculation of max_steps
    # max_steps = (args.max_epochs * (len(pairwise_train_ds) // (args.batch_size * n_gpus))) // (64 // (args.batch_size * n_gpus))
    # simplified formula below
    max_steps = args.max_epochs * (len(pairwise_train_ds) // 64)
    print(f"lr: {args.lr}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"use_scheduler: {args.use_scheduler}")
    print(f"warmup_steps: {args.warmup_steps}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_epochs: {args.max_epochs}")
    print(f"imputed max_steps: {max_steps}")

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        log_every_n_steps=10,
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        gradient_clip_val=0.2,
        logger=logger,
        default_root_dir=args.save_dir,
        callbacks=[checkpointing_cb, early_stopping_cb],
        precision="32-true",
        accumulate_grad_batches=(
            64 // (args.batch_size * n_gpus)
        ),  # original Enformer model was trained with 64 batch size using the same 0.0005 learning rate
        strategy="ddp",
    )

    model = PairwiseRegressionOnCountsWithOriginalDataJointTrainingFloatPrecision(
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_scheduler=args.use_scheduler,
        warmup_steps=args.warmup_steps,
        n_total_bins=pairwise_train_ds.get_total_n_bins(),
        checkpoint=args.enformer_checkpoint,
        state_dict_subset_prefix=args.state_dict_subset_prefix,
    )

    resume_flag = args.resume_from_checkpoint
    if args.resume_from_checkpoint:
        if len(os.listdir(ckpts_dir)) == 0:
            print("No checkpoint found to resume from. Training from scratch.")
            resume_flag = False
        else:
            previous_ckpts = os.listdir(ckpts_dir)
            # sort by step number
            print("Previous checkpoints found: ", previous_ckpts)
            previous_ckpts.sort(
                key=lambda x: int(x.split("=")[-1].split(".")[0].split("-")[0])
            )
            previous_ckpt_path = previous_ckpts[-1]

            previous_ckpt_path = os.path.join(ckpts_dir, previous_ckpt_path)
            print(f"Resuming from checkpoint: {previous_ckpt_path}")
            trainer.validate(model, dataloaders=val_dl, ckpt_path=previous_ckpt_path)
            trainer.fit(model, train_dl, val_dl, ckpt_path=previous_ckpt_path)

    if not resume_flag:
        print("Training from scratch.")
        trainer.validate(model, dataloaders=val_dl)
        trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
