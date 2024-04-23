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
    PairwiseRegressionOnCountsWithOriginalDataJointTrainingFloatPrecision,
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
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("test_data_path", type=str)
    parser.add_argument("predictions_dir", type=str)
    parser.add_argument("--seqlen", type=int, default=128 * 384)
    parser.add_argument("--batch_size", type=int, default=8)
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


def main():
    args = parse_args()
    os.makedirs(args.predictions_dir, exist_ok=True)

    test_ds = SampleH5Dataset(args.test_data_path, seqlen=args.seqlen)

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

        try:
            model = PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision(
                lr=0,
                weight_decay=0,
                use_scheduler=False,
                warmup_steps=0,
                n_total_bins=test_ds.get_total_n_bins(),
            )
            trainer.predict(
                model,
                test_dl,
                ckpt_path=args.checkpoint_path,
                return_predictions=False,
            )
            print(
                "Predicted using PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision"
            )
        except:
            try:
                model = PairwiseClassificationFloatPrecision(
                    lr=0,
                    weight_decay=0,
                    use_scheduler=False,
                    warmup_steps=0,
                    n_total_bins=test_ds.get_total_n_bins(),
                )
                trainer.predict(
                    model,
                    test_dl,
                    ckpt_path=args.checkpoint_path,
                    return_predictions=False,
                )
                print("Predicted using PairwiseClassificationFloatPrecision")
            except:
                try:
                    model = PairwiseRegressionOnCountsWithOriginalDataJointTrainingFloatPrecision(
                        lr=0,
                        weight_decay=0,
                        use_scheduler=False,
                        warmup_steps=0,
                        n_total_bins=test_ds.get_total_n_bins(),
                    )
                    trainer.predict(
                        model,
                        test_dl,
                        ckpt_path=args.checkpoint_path,
                        return_predictions=False,
                    )
                    print(
                        "Predicted using PairwiseRegressionOnCountsWithOriginalDataJointTrainingFloatPrecision"
                    )
                except:
                    try:
                        model = PairwiseRegressionFloatPrecision(
                            lr=0,
                            weight_decay=0,
                            use_scheduler=False,
                            warmup_steps=0,
                            n_total_bins=test_ds.get_total_n_bins(),
                        )
                        trainer.predict(
                            model,
                            test_dl,
                            ckpt_path=args.checkpoint_path,
                            return_predictions=False,
                        )
                        print("Predicted using PairwiseRegressionFloatPrecision")
                    except:
                        try:
                            model = SingleRegressionOnCountsFloatPrecision(
                                lr=0,
                                weight_decay=0,
                                use_scheduler=False,
                                warmup_steps=0,
                                n_total_bins=test_ds.get_total_n_bins(),
                            )
                            trainer.predict(
                                model,
                                test_dl,
                                ckpt_path=args.checkpoint_path,
                                return_predictions=False,
                            )
                            print(
                                "Predicted using SingleRegressionOnCountsFloatPrecision"
                            )
                        except:
                            try:
                                model = SingleRegressionFloatPrecision(
                                    lr=0,
                                    weight_decay=0,
                                    use_scheduler=False,
                                    warmup_steps=0,
                                    n_total_bins=test_ds.get_total_n_bins(),
                                )
                                trainer.predict(
                                    model,
                                    test_dl,
                                    ckpt_path=args.checkpoint_path,
                                    return_predictions=False,
                                )
                                print("Predicted using SingleRegressionFloatPrecision")
                            except:
                                raise ValueError(
                                    "Invalid model checkpoint. Please provide a valid model checkpoint."
                                )

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
