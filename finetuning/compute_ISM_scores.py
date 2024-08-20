import os
import pdb
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import ISMDataset
from lightning import Trainer
from models import (
    BaselineEnformer, PairwiseClassificationFloatPrecision,
    PairwiseClassificationWithOriginalDataJointTrainingFloatPrecision,
    PairwiseRegressionFloatPrecision,
    PairwiseRegressionWithOriginalDataJointTrainingFloatPrecision,
    SingleRegressionFloatPrecision, SingleRegressionOnCountsFloatPrecision)
from test_models import (
    CustomWriter, find_best_checkpoint_and_verify_that_training_is_complete)
from torch.utils.data import DataLoader
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
            "baseline",
        ],
    )
    parser.add_argument("checkpoints_dir", type=str, default=None)
    parser.add_argument("--gene_name", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seqlen", type=int, default=128 * 384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_reverse_complement", action="store_true", default=False)
    parser.add_argument(
        "--proceed_even_if_training_incomplete", action="store_true", default=False
    )
    parser.add_argument("--create_best_ckpt_copy", action="store_true", default=False)
    return parser.parse_args()


def predict(model, dl, device) -> np.ndarray:
    model.eval()
    Y_all = []

    with torch.no_grad():
        for batch in tqdm(dl):
            X = batch["seq"].to(device)
            Y = model(X, no_haplotype=True).detach().cpu().numpy()
            Y_all.append(Y)
            assert len(Y) == X.shape[0]
    return np.concatenate(Y_all, axis=0)


def main():
    args = parse_args()
    os.makedirs(args.predictions_dir, exist_ok=True)

    if args.gene_name is None:
        # get list of all population-split genes
        gene_class_df = pd.read_csv(args.gene_class_path)
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
    else:
        gene_info = pd.read_csv(args.counts_path)
        gene_info = gene_info[gene_info["our_gene_name"] == args.gene_name]
        assert (
            len(gene_info) == 1
        ), "Gene not found in the counts file, or multiple genes found with the same name."
        gene_info = gene_info[["our_gene_name", "Chr", "Coord"]].reset_index(drop=True)

    # create dataset
    test_ds = ISMDataset(
        gene_info=gene_info,
        fasta_path=args.fasta_path,
        seqlen=args.seqlen,
        use_reverse_complement=args.use_reverse_complement,
    )

    # create dataloader
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # create subdir for the gene if gene_name is provided
    if args.gene_name is not None:
        gene_dir = os.path.join(args.predictions_dir, args.gene_name)
        os.makedirs(gene_dir, exist_ok=True)
        args.predictions_dir = gene_dir

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
            model = BaselineEnformer(
                n_total_bins=(args.seqlen // 128),
            )
            print("Predicting using BaselineEnformer")
        else:
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
    genes = []
    positions = []
    nucleotides = []
    is_refs = []
    reverse_complements = []
    idxs = []
    batch_indices = []
    for i in range(n_gpus):
        p = torch.load(os.path.join(args.predictions_dir, f"predictions_{i}.pt"))
        # predictions
        p_yhat = np.concatenate([batch["Y_hat"] for batch in p])
        preds.append(p_yhat)
        # genes
        genes.append(np.concatenate([batch["gene"] for batch in p]))
        # positions
        positions.append(np.concatenate([batch["position"] for batch in p]))
        # nucleotides
        nucleotides.append(np.concatenate([batch["nucleotide"] for batch in p]))
        # is_ref
        is_refs.append(np.concatenate([batch["is_ref"] for batch in p]))
        # reverse_complement
        reverse_complements.append(
            np.concatenate([batch["reverse_complement"] for batch in p])
        )
        # idxs
        idxs.append(np.concatenate([batch["idx"] for batch in p]))

        bi = torch.load(os.path.join(args.predictions_dir, f"batch_indices_{i}.pt"))[0]
        bi = np.concatenate([inds for inds in bi])
        batch_indices.append(bi)

    preds = np.concatenate(preds, axis=0)
    genes = np.concatenate(genes, axis=0)
    positions = np.concatenate(positions, axis=0)
    nucleotides = np.concatenate(nucleotides, axis=0)
    is_refs = np.concatenate(is_refs, axis=0)
    reverse_complements = np.concatenate(reverse_complements, axis=0)
    idxs = np.concatenate(idxs, axis=0)
    batch_indices = np.concatenate(batch_indices, axis=0)

    # sort the predictions, genes, positions, nucleotides, is_refs, reverse_complements, idxs and batch_indices based on the original order
    sorted_idxs = np.argsort(batch_indices)
    idxs = idxs[sorted_idxs]
    # assert that idxs are sorted in ascending order
    assert np.all(np.diff(idxs) >= 0)
    # shape of each of the following should be (n_genes, n_positions, n_nucleotides, num_strands)
    preds = preds[sorted_idxs].reshape(
        -1, args.seqlen, 4, (1 + args.use_reverse_complement)
    )
    genes = genes[sorted_idxs].reshape(
        -1, args.seqlen, 4, (1 + args.use_reverse_complement)
    )
    positions = positions[sorted_idxs].reshape(
        -1, args.seqlen, 4, (1 + args.use_reverse_complement)
    )
    nucleotides = nucleotides[sorted_idxs].reshape(
        -1, args.seqlen, 4, (1 + args.use_reverse_complement)
    )
    is_refs = is_refs[sorted_idxs].reshape(
        -1, args.seqlen, 4, (1 + args.use_reverse_complement)
    )
    reverse_complements = reverse_complements[sorted_idxs].reshape(
        -1, args.seqlen, 4, (1 + args.use_reverse_complement)
    )

    # now average the predictions for forward and reverse strands
    preds = np.mean(preds, axis=-1)  # shape: (n_genes, n_positions, n_nucleotides)
    genes = genes[:, 0, 0, 0]  # shape: (n_genes,)
    positions = positions[:, :, 0, 0]  # shape: (n_genes, n_positions)
    nucleotides = nucleotides[
        :, :, :, 0
    ]  # shape: (n_genes, n_positions, n_nucleotides)
    is_refs = is_refs[:, :, :, 0]  # shape: (n_genes, n_positions, n_nucleotides)

    # now compute the ISM scores with respect to the reference nucleotide at each position
    ref_nucleotide_idxs = np.argmax(is_refs, axis=-1)  # shape: (n_genes, n_positions)
    ref_nucleotide_preds = np.array(
        [
            preds[i, j, ref_nucleotide_idxs[i, j]]
            for i in range(preds.shape[0])
            for j in range(preds.shape[1])
        ]
    ).reshape(
        preds.shape[:2]
    )  # shape: (n_genes, n_positions)
    ism_scores = (
        preds - ref_nucleotide_preds[:, :, np.newaxis]
    )  # shape: (n_genes, n_positions, n_nucleotides)

    # save the ISM scores
    ism_output_path = os.path.join(args.predictions_dir, "ism_scores.npz")
    np.savez(
        ism_output_path,
        ism_scores=ism_scores,
        genes=genes,
        preds=preds,
        is_refs=is_refs,
    )


if __name__ == "__main__":
    main()
