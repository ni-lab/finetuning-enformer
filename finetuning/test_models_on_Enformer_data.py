import os
import pdb
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import pandas as pd
import torch
from datasets import EnformerDataset
from lightning import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from models import (
    BaselineEnformer, PairwiseClassificationFloatPrecision,
    PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision,
    PairwiseRegressionFloatPrecision,
    PairwiseRegressionWithMalinoisMPRAJointTrainingFloatPrecision,
    PairwiseRegressionWithOriginalDataJointTrainingFloatPrecision,
    SingleRegressionFloatPrecision, SingleRegressionOnCountsFloatPrecision)
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from tqdm import tqdm


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions,
            os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"),
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("enformer_data_dir", type=str)
    parser.add_argument(
        "species", type=str, choices=["human", "mouse"], default="human"
    )
    parser.add_argument(
        "split", type=str, choices=["train", "val", "test"], default="test"
    )
    parser.add_argument("predictions_dir", type=str)
    parser.add_argument(
        "model_type",
        type=str,
        choices=[
            "baseline",
            "single_regression",
            "single_regression_counts",
            "regression",
            "joint_regression",
            "classification",
            "joint_classification",
            "joint_regression_with_Malinois_MPRA",
        ],
    )
    parser.add_argument("checkpoints_dir", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seqlen", type=int, default=128 * 384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--proceed_even_if_training_incomplete", action="store_true", default=False
    )
    parser.add_argument("--create_best_ckpt_copy", action="store_true", default=False)
    parser.add_argument("--max_train_epochs", type=int, default=50)
    parser.add_argument(
        "--add_gaussian_noise_to_pretrained_weights",
        action=BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gaussian_noise_std_multiplier", type=float, default=1)
    parser.add_argument("--seed_for_noise", type=int, default=97)
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

        elif task == "finetune_on_Enformer_data":
            # names are of the form "epoch={epoch}-step={step}-val_r2={val/human_r2_score/dataloader_idx_0:.4f}"
            ckpt_metric = f.split("val_r2=")[1].split(".ckpt")[0]
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
    os.makedirs(args.predictions_dir, exist_ok=True)

    test_ds = EnformerDataset(
        args.enformer_data_dir,
        species=args.species,
        split=args.split,
        reverse_complement=False,
        random_shift=False,
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False
    )

    # Predict on test sample sequences

    # get number of gpus
    n_gpus = torch.cuda.device_count()
    # if all predictions exist, skip the prediction step
    if all(
        [
            os.path.exists(os.path.join(args.predictions_dir, f"predictions_{i}.pt"))
            for i in range(n_gpus)
        ]
    ):
        print("Predictions already exist, skipping prediction step.")
    else:
        os.environ["SLURM_JOB_NAME"] = "interactive"
        print(f"Number of GPUs: {n_gpus}")
        pred_writer = CustomWriter(
            output_dir=args.predictions_dir, write_interval="epoch"
        )
        trainer = Trainer(
            accelerator="gpu",
            devices="auto",
            precision="32-true",
            strategy="ddp",
            callbacks=[pred_writer],
        )

        if args.model_type == "baseline":
            if args.add_gaussian_noise_to_pretrained_weights:
                np.random.seed(args.seed_for_noise)
                torch.manual_seed(args.seed_for_noise)
            model = BaselineEnformer(
                n_total_bins=(args.seqlen // 128),
                add_gaussian_noise_to_pretrained_weights=args.add_gaussian_noise_to_pretrained_weights,
                gaussian_noise_std_multiplier=args.gaussian_noise_std_multiplier,
            )
            if args.add_gaussian_noise_to_pretrained_weights:
                print(
                    f"Predicting using BaselineEnformer with Gaussian noise std multiplier {args.gaussian_noise_std_multiplier} and seed {args.seed_for_noise}"
                )
            else:
                print("Predicting using BaselineEnformer")

            trainer.predict(
                model,
                test_dl,
                return_predictions=False,
            )
        else:
            # find the best checkpoint and verify that the training is complete
            task = (
                "regression"
                if "regression" in args.model_type
                else (
                    "finetune_on_Enformer_data"
                    if "_on_enformer_data" in args.model_type
                    else "classification"
                )
            )
            best_ckpt_path = find_best_checkpoint_and_verify_that_training_is_complete(
                args.checkpoints_dir,
                task,
                args.patience,
                args.max_train_epochs,
                args.proceed_even_if_training_incomplete,
                args.create_best_ckpt_copy,
            )

            if args.model_type == "single_regression":
                model = SingleRegressionFloatPrecision(
                    lr=0,
                    weight_decay=0,
                    use_scheduler=False,
                    warmup_steps=0,
                    n_total_bins=(args.seqlen // 128),
                )
                print("Predicting using SingleRegressionFloatPrecision")
            elif args.model_type == "single_regression_counts":
                model = SingleRegressionOnCountsFloatPrecision(
                    lr=0,
                    weight_decay=0,
                    use_scheduler=False,
                    warmup_steps=0,
                    n_total_bins=(args.seqlen // 128),
                )
                print("Predicting using SingleRegressionOnCountsFloatPrecision")
            elif args.model_type == "regression":
                model = PairwiseRegressionFloatPrecision(
                    lr=0,
                    weight_decay=0,
                    use_scheduler=False,
                    warmup_steps=0,
                    n_total_bins=(args.seqlen // 128),
                )
                print("Predicting using PairwiseRegressionFloatPrecision")
            elif args.model_type == "joint_regression":
                model = PairwiseRegressionWithOriginalDataJointTrainingFloatPrecision(
                    lr=0,
                    weight_decay=0,
                    use_scheduler=False,
                    warmup_steps=0,
                    n_total_bins=(args.seqlen // 128),
                )
                print(
                    "Predicting using PairwiseRegressionWithOriginalDataJointTrainingFloatPrecision"
                )
            elif args.model_type == "classification":
                model = PairwiseClassificationFloatPrecision(
                    lr=0,
                    weight_decay=0,
                    use_scheduler=False,
                    warmup_steps=0,
                    n_total_bins=(args.seqlen // 128),
                )
                print("Predicting using PairwiseClassificationFloatPrecision")
            elif args.model_type == "joint_classification":
                model = (
                    PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision(
                        lr=0,
                        weight_decay=0,
                        use_scheduler=False,
                        warmup_steps=0,
                        n_total_bins=(args.seqlen // 128),
                    )
                )
                print(
                    "Predicting using PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision"
                )
            elif args.model_type == "joint_regression_with_Malinois_MPRA":
                model = PairwiseRegressionWithMalinoisMPRAJointTrainingFloatPrecision(
                    lr=0,
                    weight_decay=0,
                    use_scheduler=False,
                    warmup_steps=0,
                    n_total_bins=test_ds.get_total_n_bins(),
                    n_total_bins_malinois=2,
                    malinois_num_cells=5,
                )
                print(
                    "Predicting using PairwiseRegressionWithMalinoisMPRAJointTrainingFloatPrecision"
                )
            else:
                raise ValueError(
                    "Invalid model type. Please provide a valid model type."
                )

            trainer.predict(
                model,
                test_dl,
                ckpt_path=best_ckpt_path,
                return_predictions=False,
            )
        print("Done predicting.")

    # read predictions from the files and concatenate them
    # only the first rank process will read the predictions and concatenate them
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
        else:
            # wait for all processes to finish writing the predictions
            torch.distributed.barrier()

    preds = []
    targets = []
    for i in range(n_gpus):
        p = torch.load(os.path.join(args.predictions_dir, f"predictions_{i}.pt"))
        p_yhat = np.concatenate([batch["Y_hat"] for batch in p])
        p_y = np.concatenate([batch["Y"] for batch in p])
        preds.append(p_yhat)
        targets.append(p_y)

    test_preds = np.concatenate(preds, axis=0)
    test_targets = np.concatenate(targets, axis=0)

    # reshape to make metrics calculation easier
    test_preds = test_preds.reshape(-1, test_preds.shape[-1])
    test_targets = test_targets.reshape(-1, test_targets.shape[-1])

    # generate summary df containing pearson correlation, spearman correlation, and R2 score
    target_names = pd.read_csv(f"finetuning/data/targets_{args.species}.txt", sep="\t")[
        "description"
    ].values
    assert len(target_names) == test_targets.shape[1]

    summary_df = {}
    summary_df["track_name"] = target_names
    summary_df["pearson_corr"] = [
        pearsonr(test_targets[:, i], test_preds[:, i])[0]
        for i in range(test_targets.shape[1])
    ]
    summary_df["spearman_corr"] = [
        spearmanr(test_targets[:, i], test_preds[:, i])[0]
        for i in range(test_targets.shape[1])
    ]
    summary_df["r2_score"] = [
        r2_score(test_targets[:, i], test_preds[:, i])
        for i in range(test_targets.shape[1])
    ]
    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(os.path.join(args.predictions_dir, "summary_df.csv"), index=False)

    # save the predictions and targets
    np.savez_compressed(
        os.path.join(args.predictions_dir, "predictions.npz"),
        preds=test_preds,
        targets=test_targets,
    )

    print("Done saving predictions and targets.")


if __name__ == "__main__":
    main()
