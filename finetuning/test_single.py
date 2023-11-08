import os
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import RefDataset, SampleDataset
from models import *
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("ref_data_path", type=str)
    parser.add_argument("test_data_path", type=str)
    parser.add_argument("predictions_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
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

    ref_ds = RefDataset(args.ref_data_path)
    test_ds = SampleDataset(args.test_data_path)
    ref_dl = torch.utils.data.DataLoader(
        ref_ds, batch_size=args.batch_size, shuffle=False, num_workers=1
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    device = torch.device("cuda")
    model = SingleFinetuned.load_from_checkpoint(args.checkpoint_path).to(device)

    # Predict on reference sequences
    ref_preds = predict(model, ref_dl, device)
    assert ref_preds.size == ref_ds.genes.size
    ref_output_path = os.path.join(args.predictions_dir, "ref_preds.npz")
    np.savez(ref_output_path, preds=ref_preds, genes=ref_ds.genes)

    # Predict on test sample sequences
    test_preds = predict(model, test_dl, device)
    assert test_preds.size == test_ds.genes.size
    test_output_path = os.path.join(args.predictions_dir, "test_preds.npz")
    np.savez(
        test_output_path, preds=test_preds, genes=test_ds.genes, samples=test_ds.samples
    )


if __name__ == "__main__":
    main()
