import os
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import torch
from datasets import (EnformerDataset, PairwiseRegressionH5Dataset,
                      PairwiseRegressionH5DatasetDynamicSampling)
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from models import PairwiseRegressionFloatPrecision

np.random.seed(97)
torch.manual_seed(97)
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("enformer_data_path", type=str)
    parser.add_argument("run_name", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("base_run_name", type=str)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--reverse_complement_prob", type=float, default=0.5)
    parser.add_argument(
        "--do_not_random_shift", action=BooleanOptionalAction, default=False
    )
    parser.add_argument("--random_shift_max", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--enformer_checkpoint", type=str, default=None)
    parser.add_argument("--state_dict_subset_prefix", type=str, default=None)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument(
        "--resume_from_checkpoint", action=BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def find_best_checkpoint_and_verify_that_training_is_complete(
    checkpoint_dir,
    task,
    patience=5,
    max_train_epochs=50,
    proceed_even_if_training_incomplete=False,
    create_best_ckpt_copy=False,
):
    """
    Find the best checkpoint in the directory and verify that the training is complete.
    Verfication is done by checking if there are at least `patience` number of checkpoints with worse metrics than the best checkpoint.
    Args:
        checkpoint_dir: Directory containing the checkpoints.
        task: Task for which the checkpoints were saved. Can be one of "classification" or "regression". Used to determine the metric to check. For classification, it is "val_acc", and for regression, it is "val_loss".
        patience: Patience for checking if the training is complete.
        max_train_epochs: Maximum number of training epochs
        proceed_even_if_training_incomplete: If True, the function will not raise an error if the training is not complete.
        create_best_ckpt_copy: If True, the function will create a copy of the best checkpoint in the same directory with the name "best.ckpt".
    """
    # find the best checkpoint
    best_checkpoint = None
    best_metric = None
    best_metric_epoch = None
    max_epoch = -1  # epoch number of the last checkpoint
    for f in os.listdir(checkpoint_dir):
        if not f.endswith(".ckpt"):
            continue

        if f == "best.ckpt":
            print(
                "WARNING: Found a file named 'best.ckpt' in the directory. Skipping it. It will be overwritten if create_best_ckpt_copy is set to True."
            )
            continue

        max_epoch = max(max_epoch, int(f.split("epoch=")[1].split("-")[0]))
        ckpt_metric = None
        if task == "classification":
            # names are of the form "epoch={epoch}-step={step}-val_loss={classification_loss:.4f}-val_acc={classification_accuracy:.4f}.ckpt"
            # split on "-" if there are version numbers in the file name (they are added at the end)
            ckpt_metric = f.split("val_acc=")[1].split(".ckpt")[0].split("-")[0]
            ckpt_metric = float(ckpt_metric)

            if best_metric is None or ckpt_metric >= best_metric:
                if best_metric is not None and ckpt_metric == best_metric:
                    # open the ckpt files to compare the exact metric values
                    best_ckpt_so_far = torch.load(
                        os.path.join(checkpoint_dir, best_checkpoint),
                        map_location="cpu",
                    )
                    ckpt = torch.load(
                        os.path.join(checkpoint_dir, f), map_location="cpu"
                    )

                    check = False
                    for key in ckpt["callbacks"].keys():
                        if key.startswith("ModelCheckpoint"):
                            print(
                                f"Using scores from ckpts to compare the following ckpt files: {best_checkpoint} and {f}"
                            )
                            ckpt_metric = ckpt["callbacks"][key]["current_score"]
                            best_metric = best_ckpt_so_far["callbacks"][key][
                                "current_score"
                            ]
                            if ckpt_metric > best_metric:
                                check = True
                            break

                    if not check:
                        continue

                best_metric = ckpt_metric
                best_checkpoint = f
                best_metric_epoch = int(f.split("epoch=")[1].split("-")[0])

        elif task == "regression":
            # names are of the form "epoch={epoch}-step={step}-val_loss={regression_loss:.4f}.ckpt"
            # or "epoch={epoch}-step={step}-val_loss={regression_loss:.4f}-val_r2_score={val/r2_score:.4f}"
            # split on "-" if there are version numbers in the file name (they are added at the end)
            ckpt_metric = f.split("val_loss=")[1].split(".ckpt")[0].split("-")[0]
            ckpt_metric = float(ckpt_metric)

            if best_metric is None or ckpt_metric <= best_metric:
                if best_metric is not None and ckpt_metric == best_metric:
                    # open the ckpt files to compare the exact metric values
                    best_ckpt_so_far = torch.load(
                        os.path.join(checkpoint_dir, best_checkpoint),
                        map_location="cpu",
                    )
                    ckpt = torch.load(
                        os.path.join(checkpoint_dir, f), map_location="cpu"
                    )

                    check = False
                    for key in ckpt["callbacks"].keys():
                        if key.startswith("ModelCheckpoint"):
                            print(
                                f"Using scores from ckpts to compare the following ckpt files: {best_checkpoint} and {f}"
                            )
                            ckpt_metric = ckpt["callbacks"][key]["current_score"]
                            best_metric = best_ckpt_so_far["callbacks"][key][
                                "current_score"
                            ]
                            if ckpt_metric < best_metric:
                                check = True
                            break

                    if not check:
                        continue

                best_metric = ckpt_metric
                best_checkpoint = f
                best_metric_epoch = int(f.split("epoch=")[1].split("-")[0])

    # check if the training is complete
    if best_checkpoint is None:
        raise ValueError("No checkpoint found in the directory.")
    if max_epoch - best_metric_epoch < patience:
        if max_epoch == (max_train_epochs - 1):
            print("WARNING: Max training epochs completed, so patience ignored")
        else:
            if not proceed_even_if_training_incomplete:
                raise ValueError(
                    f"Training may not be complete. Current best checkpoint is from epoch {best_metric_epoch} and the last checkpoint is from epoch {max_epoch}."
                )
            else:
                print(
                    "WARNING: Training may not be complete. Current best checkpoint is from epoch",
                    best_metric_epoch,
                    "and the last checkpoint is from epoch",
                    max_epoch,
                )

    # create a copy of the best checkpoint
    if create_best_ckpt_copy:
        best_ckpt_path = os.path.join(checkpoint_dir, best_checkpoint)
        best_ckpt_copy_path = os.path.join(checkpoint_dir, "best.ckpt")
        os.system(f"cp {best_ckpt_path} {best_ckpt_copy_path}")
        print(f"Created a copy of the best checkpoint at {best_ckpt_copy_path}")

    return os.path.join(checkpoint_dir, best_checkpoint)


def main():
    args = parse_args()

    if args.do_not_random_shift:
        print("Not using random shift.")
        args.random_shift_max = 0

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
                human_enformer_train_ds, batch_size=args.batch_size
            ),
            torch.utils.data.DataLoader(
                mouse_enformer_train_ds, batch_size=args.batch_size
            ),
        ],
        mode="max_size_cycle",
    )

    val_dl = [
        torch.utils.data.DataLoader(human_enformer_val_ds, batch_size=args.batch_size),
        torch.utils.data.DataLoader(mouse_enformer_val_ds, batch_size=args.batch_size),
    ]

    run_suffix = f"_data_seed_{args.data_seed}_lr_{args.lr}_wd_{args.weight_decay}_rcprob_{args.reverse_complement_prob}_rsmax_{args.random_shift_max}"

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
        filename="epoch={epoch}-step={step}-val_loss={val/human_r2_score/dataloader_idx_0:.4f}",
        monitor="val/human_r2_score/dataloader_idx_0",
        mode="max",
        save_top_k=-1,
        auto_insert_metric_name=False,
    )

    early_stopping_cb = EarlyStopping(
        monitor="val/human_r2_score/dataloader_idx_0",
        mode="max",
        patience=5,
    )

    os.environ["SLURM_JOB_NAME"] = "interactive"
    # get number of gpus
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")

    # print hyperparameters
    print(f"lr: {args.lr}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_epochs: {args.max_epochs}")

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        log_every_n_steps=10,
        max_epochs=args.max_epochs,
        gradient_clip_val=0.2,
        logger=logger,
        default_root_dir=args.save_dir,
        callbacks=[checkpointing_cb, early_stopping_cb]
        if not args.use_random_init
        else [
            checkpointing_cb
        ],  # don't use early stopping if using random init to aid convergence
        precision="32-true",
        accumulate_grad_batches=(
            64 // (args.batch_size * n_gpus)
        ),  # original Enformer model was trained with 64 batch size using the same 0.0005 learning rate
        strategy="ddp_find_unused_parameters_falso",
    )

    model = PairwiseRegressionFloatPrecision(
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_scheduler=args.use_scheduler,
        warmup_steps=args.warmup_steps,
        n_total_bins=384,
        checkpoint=args.enformer_checkpoint,
        state_dict_subset_prefix=args.state_dict_subset_prefix,
        use_random_init=args.use_random_init,
        add_gaussian_noise_to_pretrained_weights=args.add_gaussian_noise_to_pretrained_weights,
        gaussian_noise_std_multiplier=args.gaussian_noise_std_multiplier,
        freeze_cnn=args.freeze_cnn,
        freeze_transformer=args.freeze_transformer,
        finetune_enformer_output_heads_only=True,
    )

    # find best checkpoint from base finetuning run and restore those weights
    best_ckpt_path_from_base_finetuning_run = (
        find_best_checkpoint_and_verify_that_training_is_complete(
            os.path.join(args.save_dir, args.base_run_name, "checkpoints"),
            task="regression",
        )
    )
    model.load_from_checkpoint(best_ckpt_path_from_base_finetuning_run)

    # freeze all weights except those of the Enformer output heads
    ori_num_params = sum(p.numel() for p in model.parameters())
    model.freeze_all_weights_except_enformer_output_heads()
    new_num_params = sum(p.numel() for p in model.parameters())
    print(
        f"Number of parameters frozen: {ori_num_params - new_num_params} out of {ori_num_params}"
    )
    assert ori_num_params > new_num_params, "No parameters were frozen"

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
            # trainer.validate(model, dataloaders=val_dl, ckpt_path=previous_ckpt_path)
            trainer.fit(model, train_dl, val_dl, ckpt_path=previous_ckpt_path)

    if not resume_flag:
        print("Training from scratch.")
        trainer.validate(model, dataloaders=val_dl)
        trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
