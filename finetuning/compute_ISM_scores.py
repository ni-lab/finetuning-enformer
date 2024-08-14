import os
import pdb
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import SampleH5Dataset
from enformer_pytorch.data import str_to_one_hot
from lightning import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from models import (
    PairwiseClassificationFloatPrecision,
    PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision,
    PairwiseRegressionFloatPrecision,
    PairwiseRegressionWithOriginalDataJointTrainingFloatPrecision,
    SingleRegressionFloatPrecision, SingleRegressionOnCountsFloatPrecision)
from pyfaidx import Fasta
from test_models import (
    CustomWriter, find_best_checkpoint_and_verify_that_training_is_complete,
    predict)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("counts_path", type=str)  # path to the counts file
    parser.add_argument("gene_class_path", type=str)  # path to the gene class file
    parser.add_argument("fasta_path", type=str)  # path to the reference genome
    parser.add_argument("predictions_dir", type=str)
    parser.add_argument(
        "model_type",
        type=str,
        choices=[
            "single_regression",
            "single_regression_counts",
            "regression",
            "joint_regression",
            "classification",
            "joint_classification",
        ],
    )
    parser.add_argument("checkpoints_dir", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seqlen", type=int, default=128 * 384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_reverse_complement", action="store_true", default=False)
    parser.add_argument(
        "--proceed_even_if_training_incomplete", action="store_true", default=False
    )
    parser.add_argument("--create_best_ckpt_copy", action="store_true", default=False)
    return parser.parse_args()


class ISMDataset(Dataset):
    """
    Takes in the gene info and reference genome, and generates every possible single nucleotide variant for each gene
    """

    def __init__(self, gene_info, fasta_path, seqlen, use_reverse_complement):
        self.gene_info = (
            gene_info  # must contain columns "our_gene_name", "Chr", "Coord"
        )
        self.fasta = Fasta(fasta_path)
        self.seqlen = seqlen
        self.use_reverse_complement = use_reverse_complement

        # get the reference sequence for each gene from the fasta file
        ref_seqs = []
        print("Getting reference sequences for all genes")
        for i in tqdm(range(len(self.gene_info))):
            row = self.gene_info.iloc[i]
            gene = row["our_gene_name"]
            if pd.isna(gene):
                raise ValueError("Gene name is missing")
            chrom = row["Chr"]
            bp_start = row["Coord"] - self.seqlen // 2
            bp_end = bp_start + self.seqlen - 1
            ref_seq = self.fasta[f"chr{chrom}"][bp_start - 1 : bp_end].seq.upper()
            assert len(ref_seq) == self.seqlen
            ref_seqs.append(ref_seq)
        self.gene_info["ref_seq"] = ref_seqs

    def __len__(self):
        return (
            len(self.ref_seqs)
            * self.seqlen
            * 4
            * (1 + int(self.use_reverse_complement))
        )

    def __getitem__(self, idx):
        gene_idx = idx // (self.seqlen * 4 * (1 + int(self.use_reverse_complement)))
        position_nucleotide_rc_offset = idx % (
            self.seqlen * 4 * (1 + int(self.use_reverse_complement))
        )
        position = position_nucleotide_rc_offset // (
            4 * (1 + int(self.use_reverse_complement))
        )
        nucleotide_rc_offset = position_nucleotide_rc_offset % (
            4 * (1 + int(self.use_reverse_complement))
        )
        nucleotide = nucleotide_rc_offset // (1 + int(self.use_reverse_complement))
        rc = nucleotide_rc_offset % (1 + int(self.use_reverse_complement))

        ref_seq = self.gene_info.iloc[gene_idx]["ref_seq"]
        ref_seq = ref_seq[:position] + "ACGT"[nucleotide] + ref_seq[position + 1 :]
        if rc:
            ref_seq = ref_seq[::-1].translate(str.maketrans("ACGT", "TGCA"))

        ref_seq = str_to_one_hot(ref_seq)

        return {
            "seq": ref_seq,
            "gene": self.gene_info.iloc[gene_idx]["our_gene_name"],
            "position": position,
            "nucleotide": "ACGT"[nucleotide],
            "reverse_complement": rc,
        }


def main():
    args = parse_args()
    os.makedirs(args.predictions_dir, exist_ok=True)

    # get list of all population-split genes
    gene_class_df = pd.read_csv(args.gene_class_path, sep="\t")
    population_split_genes = gene_class_df[
        gene_class_df["class"] == "yri_split"
    ].reset_index(drop=True)
    print(f"Number of population-split genes: {len(population_split_genes)}")

    # get chromosome and coordinate information for each gene
    gene_info = pd.read_csv(args.counts_path)
    gene_info = (
        gene_info[["our_gene_name", "Chr", "Coord"]]
        .merge(
            population_split_genes,
            left_on="our_gene_name",
            right_on="gene",
            how="inner",
        )
        .drop(columns=["gene", "class"])
    )

    # create dataset
    test_ds = ISMDataset(
        gene_info=gene_info,
        fasta_path=args.fasta_path,
        seqlen=args.seqlen,
        use_reverse_complement=args.use_reverse_complement,
    )

    # create dataloader
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

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
            args.checkpoints_dir,
            task,
            args.patience,
            args.proceed_even_if_training_incomplete,
            args.create_best_ckpt_copy,
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
        elif args.model_type == "single_regression_counts":
            model = SingleRegressionOnCountsFloatPrecision(
                lr=0,
                weight_decay=0,
                use_scheduler=False,
                warmup_steps=0,
                n_total_bins=test_ds.get_total_n_bins(),
            )
            print("Predicting using SingleRegressionOnCountsFloatPrecision")
        elif args.model_type == "regression":
            model = PairwiseRegressionFloatPrecision(
                lr=0,
                weight_decay=0,
                use_scheduler=False,
                warmup_steps=0,
                n_total_bins=test_ds.get_total_n_bins(),
            )
            print("Predicting using PairwiseRegressionFloatPrecision")
        elif args.model_type == "joint_regression":
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
        elif args.model_type == "classification":
            model = PairwiseClassificationFloatPrecision(
                lr=0,
                weight_decay=0,
                use_scheduler=False,
                warmup_steps=0,
                n_total_bins=test_ds.get_total_n_bins(),
            )
            print("Predicting using PairwiseClassificationFloatPrecision")
        elif args.model_type == "joint_classification":
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

    # preds = []
    # true_idxs = []
    # batch_indices = []
    # for i in range(n_gpus):
    #     p = torch.load(os.path.join(args.predictions_dir, f"predictions_{i}.pt"))
    #     p_yhat = np.concatenate([batch["Y_hat"] for batch in p])
    #     preds.append(p_yhat)
    #     true_idxs.append(np.concatenate([batch["true_idx"] for batch in p]))

    #     bi = torch.load(os.path.join(args.predictions_dir, f"batch_indices_{i}.pt"))[0]
    #     bi = np.concatenate([inds for inds in bi])
    #     batch_indices.append(bi)

    # test_preds = np.concatenate(preds, axis=0)
    # true_idxs = np.concatenate(true_idxs, axis=0)
    # batch_indices = np.concatenate(batch_indices, axis=0)

    # # sort the predictions, true_idxs and batch_indices based on the original order
    # sorted_idxs = np.argsort(batch_indices)
    # test_preds = test_preds[sorted_idxs]
    # true_idxs = true_idxs[sorted_idxs]

    # # now average the predictions that have the same true index
    # unique_true_idxs = np.unique(true_idxs)
    # unique_true_idxs = np.sort(unique_true_idxs)
    # averaged_preds = []
    # for idx in unique_true_idxs:
    #     idx_mask = true_idxs == idx
    #     avg_pred = np.mean(test_preds[idx_mask], axis=0)
    #     averaged_preds.append(avg_pred)
    # test_preds = np.array(averaged_preds)

    # assert test_preds.size == test_ds.genes.size
    # test_output_path = os.path.join(args.predictions_dir, "test_preds.npz")
    # np.savez(
    #     test_output_path, preds=test_preds, genes=test_ds.genes, samples=test_ds.samples
    # )


if __name__ == "__main__":
    main()
