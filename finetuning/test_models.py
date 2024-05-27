import os
import pdb
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import SampleH5Dataset
from lightning import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from models import (
    PairwiseClassificationFloatPrecision,
    PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision,
    PairwiseRegressionFloatPrecision,
    PairwiseRegressionWithOriginalDataJointTrainingFloatPrecision,
    SingleRegressionFloatPrecision, SingleRegressionOnCountsFloatPrecision)
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

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(
            batch_indices,
            os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"),
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("test_data_path", type=str)
    parser.add_argument("predictions_dir", type=str)
    parser.add_argument(
        "model_type",
        type=str,
        choices=[
            "single_regression",
            "single_regression_on_counts",
            "pairwise_regression",
            "pairwise_regression_joint_training",
            "pairwise_classification",
            "pairwise_classification_joint_training",
        ],
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seqlen", type=int, default=128 * 384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_reverse_complement", action="store_true", default=False)
    return parser.parse_args()


def predict(model, dl, device) -> np.ndarray:
    model.eval()
    Y_all = []

    with torch.no_grad():
        for batch in tqdm(dl):
            X = batch["seq"].to(device)
            Y = model(X).detach().cpu().numpy()
            Y_all.append(Y)
    return np.concatenate(Y_all, axis=0)


def find_best_checkpoint_and_verify_that_training_is_complete(
    checkpoint_dir, task, patience=5
):
    """
    Find the best checkpoint in the directory and verify that the training is complete.
    Verfication is done by checking if there are at least `patience` number of checkpoints with worse metrics than the best checkpoint.
    Args:
        checkpoint_dir: Directory containing the checkpoints.
        task: Task for which the checkpoints were saved. Can be one of "classification" or "regression". Used to determine the metric to check. For classification, it is "val_acc", and for regression, it is "val_loss".
        patience: Patience for checking if the training is complete.
    """
    # find the best checkpoint
    best_checkpoint = None
    best_metric = None
    best_metric_epoch = None
    max_epoch = -1  # epoch number of the last checkpoint
    for f in os.listdir(checkpoint_dir):
        if not f.endswith(".ckpt"):
            continue

        ckpt_metric = None
        if task == "classification":
            # names are of the form "epoch={epoch}-step={step}-val_loss={pairwise_classification_loss:.4f}-val_acc={pairwise_classification_accuracy:.4f}.ckpt"
            # split on "-" if there are version numbers in the file name (they are added at the end)
            ckpt_metric = f.split("val_acc=")[1].split(".ckpt")[0].split("-")[0]
            ckpt_metric = float(ckpt_metric)

            if best_metric is None or ckpt_metric >= best_metric:
                if best_metric is not None and ckpt_metric == best_metric:
                    ckpt_epoch = int(f.split("epoch=")[1].split("-")[0])
                    if ckpt_epoch < best_metric_epoch:
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
                    ckpt_epoch = int(f.split("epoch=")[1].split("-")[0])
                    if ckpt_epoch < best_metric_epoch:
                        continue

                best_metric = ckpt_metric
                best_checkpoint = f
                best_metric_epoch = int(f.split("epoch=")[1].split("-")[0])

        max_epoch = max(max_epoch, int(f.split("epoch=")[1].split("-")[0]))

    # check if the training is complete
    if best_checkpoint is None:
        raise ValueError("No checkpoint found in the directory.")
    if max_epoch - best_metric_epoch < patience:
        print(
            "WARNING: Training may not be complete. Current best checkpoint is from epoch",
            best_metric_epoch,
            "and the last checkpoint is from epoch",
            max_epoch,
        )

    return best_checkpoint


def main():
    args = parse_args()
    os.makedirs(args.predictions_dir, exist_ok=True)

    test_ds = SampleH5Dataset(
        args.test_data_path,
        seqlen=args.seqlen,
        return_reverse_complement=args.use_reverse_complement,
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

        # find the best checkpoint and verify that the training is complete
        task = "regression" if "regression" in args.model_type else "classification"
        best_ckpt_path = find_best_checkpoint_and_verify_that_training_is_complete(
            args.checkpoint_path, task, args.patience
        )

        if args.model_type == "single_regression":
            model = SingleRegressionFloatPrecision(
                lr=0,
                weight_decay=0,
                use_scheduler=False,
                warmup_steps=0,
                n_total_bins=test_ds.get_total_n_bins(),
            )
            print("Predicting using SingleRegressionFloatPrecision")
        elif args.model_type == "single_regression_on_counts":
            model = SingleRegressionOnCountsFloatPrecision(
                lr=0,
                weight_decay=0,
                use_scheduler=False,
                warmup_steps=0,
                n_total_bins=test_ds.get_total_n_bins(),
            )
            print("Predicting using SingleRegressionOnCountsFloatPrecision")
        elif args.model_type == "pairwise_regression":
            model = PairwiseRegressionFloatPrecision(
                lr=0,
                weight_decay=0,
                use_scheduler=False,
                warmup_steps=0,
                n_total_bins=test_ds.get_total_n_bins(),
            )
            print("Predicting using PairwiseRegressionFloatPrecision")
        elif args.model_type == "pairwise_regression_joint_training":
            model = PairwiseRegressionWithOriginalDataJointTrainingFloatPrecision(
                lr=0,
                weight_decay=0,
                use_scheduler=False,
                warmup_steps=0,
                n_total_bins=test_ds.get_total_n_bins(),
            )
            print(
                "Predicting using PairwiseRegressionWithOriginalDataJointTrainingFloatPrecision"
            )
        elif args.model_type == "pairwise_classification":
            model = PairwiseClassificationFloatPrecision(
                lr=0,
                weight_decay=0,
                use_scheduler=False,
                warmup_steps=0,
                n_total_bins=test_ds.get_total_n_bins(),
            )
            print("Predicting using PairwiseClassificationFloatPrecision")
        elif args.model_type == "pairwise_classification_joint_training":
            model = PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision(
                lr=0,
                weight_decay=0,
                use_scheduler=False,
                warmup_steps=0,
                n_total_bins=test_ds.get_total_n_bins(),
            )
            print(
                "Predicting using PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision"
            )
        else:
            raise ValueError("Invalid model type. Please provide a valid model type.")

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
    true_idxs = []
    batch_indices = []
    for i in range(n_gpus):
        p = torch.load(os.path.join(args.predictions_dir, f"predictions_{i}.pt"))
        p_yhat = np.concatenate([batch["Y_hat"] for batch in p])
        preds.append(p_yhat)
        true_idxs.append(np.concatenate([batch["true_idx"] for batch in p]))

        bi = torch.load(os.path.join(args.predictions_dir, f"batch_indices_{i}.pt"))[0]
        bi = np.concatenate([inds for inds in bi])
        batch_indices.append(bi)

    test_preds = np.concatenate(preds, axis=0)
    true_idxs = np.concatenate(true_idxs, axis=0)
    batch_indices = np.concatenate(batch_indices, axis=0)

    # sort the predictions, true_idxs and batch_indices based on the original order
    sorted_idxs = np.argsort(batch_indices)
    test_preds = test_preds[sorted_idxs]
    true_idxs = true_idxs[sorted_idxs]

    # now average the predictions that have the same true index
    unique_true_idxs = np.unique(true_idxs)
    unique_true_idxs = np.sort(unique_true_idxs)
    averaged_preds = []
    for idx in unique_true_idxs:
        idx_mask = true_idxs == idx
        avg_pred = np.mean(test_preds[idx_mask], axis=0)
        averaged_preds.append(avg_pred)
    test_preds = np.array(averaged_preds)

    assert test_preds.size == test_ds.genes.size
    test_output_path = os.path.join(args.predictions_dir, "test_preds.npz")
    np.savez(
        test_output_path, preds=test_preds, genes=test_ds.genes, samples=test_ds.samples
    )


if __name__ == "__main__":
    main()
