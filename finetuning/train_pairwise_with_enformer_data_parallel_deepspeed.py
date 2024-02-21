import os
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from datasets import EnformerDataset, PairwiseDataset
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from models import PairwiseWithOriginalDataJointTraining

torch.manual_seed(97)
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("train_data_path", type=str)
    parser.add_argument("val_data_path", type=str)
    parser.add_argument("enformer_data_path", type=str)
    parser.add_argument("run_name", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze_at_epoch", type=int, default=1)
    parser.add_argument("--initial_denom_lr", type=float, default=1.0)
    parser.add_argument("--train_bn", action=BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_n_pairs", type=int, default=100_000)
    parser.add_argument("--val_n_pairs", type=int, default=5_000)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=250000)
    parser.add_argument("--enformer_checkpoint", type=str, default=None)
    parser.add_argument("--state_dict_subset_prefix", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    pairwise_train_ds = PairwiseDataset(
        args.train_data_path, n_pairs=args.train_n_pairs
    )
    pairwise_val_ds = PairwiseDataset(args.val_data_path, n_pairs=args.val_n_pairs)

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
                pairwise_train_ds, batch_size=args.batch_size * 4, shuffle=True
            ),
            torch.utils.data.DataLoader(
                human_enformer_train_ds, batch_size=args.batch_size
            ),
            torch.utils.data.DataLoader(
                mouse_enformer_train_ds, batch_size=args.batch_size
            ),
        ],
        mode="min_size",
    )

    val_dl = [
        torch.utils.data.DataLoader(
            pairwise_val_ds, batch_size=args.batch_size * 4, shuffle=False
        ),
        torch.utils.data.DataLoader(human_enformer_val_ds, batch_size=args.batch_size),
        torch.utils.data.DataLoader(mouse_enformer_val_ds, batch_size=args.batch_size),
    ]

    logger = WandbLogger(
        project="enformer-finetune", name=args.run_name, save_dir=args.save_dir
    )

    checkpointing_cb = ModelCheckpoint(
        monitor="val/pairwise_mse_loss/dataloader_idx_0",
        mode="min",
        save_top_k=1,
    )

    early_stopping_cb = EarlyStopping(
        monitor="val/pairwise_mse_loss/dataloader_idx_0", mode="min", patience=5
    )

    os.environ["SLURM_JOB_NAME"] = "interactive"
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        log_every_n_steps=10,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        gradient_clip_val=1.0,
        logger=logger,
        default_root_dir=args.save_dir,
        callbacks=[checkpointing_cb, early_stopping_cb],
        precision="16-mixed",
        accumulate_grad_batches=(64 // (args.batch_size * 8)),
        strategy=DeepSpeedStrategy(
            logging_batch_size_per_gpu=args.batch_size,
            stage=2,
            offload_optimizer=True,
            offload_parameters=True,
            min_loss_scale=0,
        ),
    )

    model = PairwiseWithOriginalDataJointTraining(
        lr=args.lr,
        n_total_bins=pairwise_train_ds.get_total_n_bins(),
        checkpoint=args.enformer_checkpoint,
        state_dict_subset_prefix=args.state_dict_subset_prefix,
    )
    trainer.validate(model, dataloaders=val_dl)
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
