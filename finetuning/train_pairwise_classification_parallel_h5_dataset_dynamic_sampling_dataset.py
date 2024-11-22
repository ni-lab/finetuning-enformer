import os
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import torch
from datasets import (PairwiseClassificationH5Dataset,
                      PairwiseClassificationH5DatasetDynamicSampling)
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from models import PairwiseClassificationFloatPrecision

np.random.seed(97)
torch.manual_seed(97)
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("train_data_path", type=str)
    parser.add_argument("val_data_path", type=str)
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
    parser.add_argument("--reverse_complement_prob", type=float, default=0.5)
    parser.add_argument(
        "--do_not_random_shift", action=BooleanOptionalAction, default=False
    )
    parser.add_argument("--random_shift_max", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--enformer_checkpoint", type=str, default=None)
    parser.add_argument("--state_dict_subset_prefix", type=str, default=None)
    parser.add_argument(
        "--use_random_init", action=BooleanOptionalAction, default=False
    )
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--train_set_subsample_ratio", type=float, default=1.0)
    parser.add_argument(
        "--resume_from_checkpoint", action=BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.do_not_random_shift:
        print("Not using random shift.")
        args.random_shift_max = 0

    pairwise_train_ds = PairwiseClassificationH5DatasetDynamicSampling(
        args.train_data_path,
        n_pairs_per_gene=args.train_n_pairs_per_gene,
        seqlen=args.seqlen,
        random_seed=args.data_seed,
        reverse_complement_prob=args.reverse_complement_prob,
        random_shift=not args.do_not_random_shift,
        random_shift_max=args.random_shift_max,
        subsample_ratio=args.train_set_subsample_ratio,
    )
    pairwise_val_ds = PairwiseClassificationH5Dataset(
        args.val_data_path,
        n_pairs_per_gene=args.val_n_pairs_per_gene,
        seqlen=args.seqlen,
        return_reverse_complement=args.reverse_complement_prob > 0.0,
        shift_max=0,
    )

    train_dl = torch.utils.data.DataLoader(
        pairwise_train_ds, batch_size=args.batch_size, shuffle=True
    )

    val_dl = torch.utils.data.DataLoader(
        pairwise_val_ds, batch_size=args.batch_size, shuffle=False
    )

    run_suffix = f"_data_seed_{args.data_seed}_lr_{args.lr}_wd_{args.weight_decay}_rcprob_{args.reverse_complement_prob}_rsmax_{args.random_shift_max}"
    if args.train_set_subsample_ratio < 1.0:
        run_suffix += f"_subsample_ratio_{args.train_set_subsample_ratio}"
    if args.use_random_init:
        run_suffix += "_random_init"

    run_save_dir = os.path.join(
        args.save_dir,
        args.run_name + run_suffix,
    )
    os.makedirs(run_save_dir, exist_ok=True)

    logs_dir = os.path.join(run_save_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    ckpts_dir = os.path.join(run_save_dir, "checkpoints")
    os.makedirs(ckpts_dir, exist_ok=True)

    logger = WandbLogger(
        project="enformer-finetune",
        name=args.run_name + run_suffix,
        save_dir=logs_dir,
    )

    checkpointing_cb = ModelCheckpoint(
        dirpath=ckpts_dir,
        filename="epoch={epoch}-step={step}-val_loss={val/pairwise_classification_loss:.4f}-val_acc={val/pairwise_classification_accuracy:.4f}",
        monitor="val/pairwise_classification_accuracy",
        mode="max",
        save_top_k=-1,
        auto_insert_metric_name=False,
    )

    early_stopping_cb = EarlyStopping(
        monitor="val/pairwise_classification_accuracy",
        mode="max",
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
        strategy="ddp_find_unused_parameters_true",
    )

    model = PairwiseClassificationFloatPrecision(
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_scheduler=args.use_scheduler,
        warmup_steps=args.warmup_steps,
        n_total_bins=pairwise_train_ds.get_total_n_bins(),
        checkpoint=args.enformer_checkpoint,
        state_dict_subset_prefix=args.state_dict_subset_prefix,
        use_random_init=args.use_random_init,
    )

    resume_flag = args.resume_from_checkpoint
    if args.resume_from_checkpoint:
        if len(os.listdir(ckpts_dir)) == 0:
            print("No checkpoint found to resume from. Training from scratch.")
            resume_flag = False
        else:
            previous_ckpts = os.listdir(ckpts_dir)
            print("Previous checkpoints found: ", previous_ckpts)
            if "best.ckpt" in previous_ckpts:
                print(
                    "Training has been completed and test script was run to generate best.ckpt, skipping training."
                )
                return

            # sort by epoch number
            print(
                "Epoch numbers: ",
                [int(x.split("-")[0].split("=")[1]) for x in previous_ckpts],
            )
            previous_ckpts.sort(key=lambda x: int(x.split("-")[0].split("=")[1]))
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
