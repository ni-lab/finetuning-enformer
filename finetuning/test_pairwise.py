import os
import pdb
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import RefDataset, SampleDataset, SampleH5Dataset
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    try:
        model = PairwiseFinetuned.load_from_checkpoint(args.checkpoint_path).to(device)
    except:
        try:
            model = PairwiseWithOriginalDataJointTrainingFloatPrecision.load_from_checkpoint(
                args.checkpoint_path
            ).to(
                device
            )
        except:
            try:
                model = PairwiseWithOriginalDataJointTrainingAndPairwiseMPRAFloatPrecision.load_from_checkpoint(
                    args.checkpoint_path
                ).to(
                    device
                )
            except:
                try:
                    model = PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision.load_from_checkpoint(
                        args.checkpoint_path
                    ).to(
                        device
                    )
                except:
                    raise ValueError(
                        "Invalid model checkpoint path - must be one of PairwiseFinetuned, PairwiseWithOriginalDataJointTrainingFloatPrecision, PairwiseWithOriginalDataJointTrainingAndPairwiseMPRAFloatPrecision, or PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision."
                    )

    if args.test_data_path.endswith(".h5"):
        test_ds = SampleH5Dataset(args.test_data_path, seqlen=args.seqlen)
    else:
        test_ds = SampleDataset(args.test_data_path)

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False
    )

    # Predict on test sample sequences
    test_preds = predict(model, test_dl, device)
    assert test_preds.size == test_ds.genes.size
    test_output_path = os.path.join(args.predictions_dir, "test_preds.npz")
    np.savez(
        test_output_path, preds=test_preds, genes=test_ds.genes, samples=test_ds.samples
    )


if __name__ == "__main__":
    main()
