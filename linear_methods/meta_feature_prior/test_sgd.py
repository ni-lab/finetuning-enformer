import os
from argparse import ArgumentParser
from collections import defaultdict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from train_sgd import Dataset, Model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--prediction_dir", type=str, default=None)
    args = parser.parse_args()

    if args.prediction_dir is None:
        grandparent_dir = os.path.dirname(os.path.dirname(args.ckpt_path))
        args.prediction_dir = os.path.join(grandparent_dir, "preds")
    return args


def main():
    args = parse_args()

    test_ds = Dataset(os.path.join(args.dataset_dir, "test.h5"))
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model.load_from_checkpoint(args.ckpt_path).to(device)
    model.eval()

    with torch.no_grad():
        preds = defaultdict(dict)  # gene -> sample -> prediction
        counts = defaultdict(dict)  # gene -> sample -> count
        for batch in tqdm(test_dl):
            gene = batch["gene"][0]
            features = batch["feature"][0].to(device)  # (n_samples, n_variants)
            Y = batch["Y"][0].to(device)  # (n_samples)
            samples = test_ds.samples[gene]  # (n_samples)
            Y_hat = model(gene, features).detach().cpu().numpy()  # (n_samples)
            assert Y.shape[0] == samples.size == Y_hat.size

            for i, sample in enumerate(samples):
                preds[gene][sample] = Y_hat[i]
                counts[gene][sample] = Y[i]

    os.makedirs(args.prediction_dir, exist_ok=True)

    genes = sorted(preds.keys())
    all_samples = list(set.union(*[set(preds[g].keys()) for g in genes]))
    preds_mtx = np.full((len(genes), len(all_samples)), np.nan)
    counts_mtx = np.full((len(genes), len(all_samples)), np.nan)

    for i, gene in enumerate(genes):
        for j, sample in enumerate(all_samples):
            preds_mtx[i, j] = preds[gene].get(sample, np.nan)
            counts_mtx[i, j] = counts[gene].get(sample, np.nan)

    preds_df = pd.DataFrame(preds_mtx, index=genes, columns=all_samples)
    counts_df = pd.DataFrame(counts_mtx, index=genes, columns=all_samples)
    ckpt_name = os.path.basename(args.ckpt_path).replace(".ckpt", "")
    preds_df.to_csv(os.path.join(args.prediction_dir, f"{ckpt_name}.preds.csv"))
    counts_df.to_csv(os.path.join(args.prediction_dir, f"{ckpt_name}.counts.csv"))


if __name__ == "__main__":
    main()
