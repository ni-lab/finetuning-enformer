import os
import pdb
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import RefDataset, SampleDataset, SampleH5Dataset
from lightning import Trainer
from models import (
    PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision,
    PairwiseFinetuned,
    PairwiseWithOriginalDataJointTrainingAndPairwiseMPRAFloatPrecision,
    PairwiseWithOriginalDataJointTrainingFloatPrecision)
from tqdm import tqdm


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

    if args.test_data_path.endswith(".h5"):
        test_ds = SampleH5Dataset(args.test_data_path, seqlen=args.seqlen)
    else:
        test_ds = SampleDataset(args.test_data_path)

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False
    )

    try:
        model = PairwiseFinetuned.load_from_checkpoint(args.checkpoint_path)
    except:
        try:
            model = PairwiseWithOriginalDataJointTrainingFloatPrecision.load_from_checkpoint(
                args.checkpoint_path
            )
        except:
            try:
                model = PairwiseWithOriginalDataJointTrainingAndPairwiseMPRAFloatPrecision.load_from_checkpoint(
                    args.checkpoint_path
                )
            except:
                try:
                    model = PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision.load_from_checkpoint(
                        args.checkpoint_path
                    )
                except:
                    raise ValueError(
                        "Invalid model checkpoint path - must be one of PairwiseFinetuned, PairwiseWithOriginalDataJointTrainingFloatPrecision, PairwiseWithOriginalDataJointTrainingAndPairwiseMPRAFloatPrecision, or PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision."
                    )

    # Predict on test sample sequences
    if isinstance(
        model, PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision
    ):  # this model has an inbuilt predict step
        os.environ["SLURM_JOB_NAME"] = "interactive"
        # get number of gpus
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {n_gpus}")
        trainer = Trainer(
            accelerator="gpu",
            devices="auto",
            precision="32-true",
            strategy="ddp",
        )

        model = PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision(
            lr=0,
            n_total_bins=test_ds.get_total_n_bins(),
        )

        predictions = trainer.predict(model, test_dl, ckpt_path=args.checkpoint_path)
        test_preds = np.concatenate(
            [pred["Y_hat"].detach().cpu().numpy() for pred in predictions]
        )
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        model.to(device)
        test_preds = predict(model, test_dl, device)

    assert test_preds.size == test_ds.genes.size
    test_output_path = os.path.join(args.predictions_dir, "test_preds.npz")
    np.savez(
        test_output_path, preds=test_preds, genes=test_ds.genes, samples=test_ds.samples
    )


if __name__ == "__main__":
    main()
