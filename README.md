# Fine-tuning sequence-to-expression models on personal genome and transcriptome data

The code base for our work on improving the performance of sequence-to-expression models for making individual-specific gene expression predictions by fine-tuning them on personal genome and transcriptome data. Please cite the following paper if you use our code:
```bibtex
@article{
    finetuning_seq2exp_models_rastogi_reddy_2024,
    author = {Rastogi, Ruchir and Reddy, Aniketh Janardhan and Chung, Ryan and Ioannidis, Nilah M.},
    title = {Fine-tuning sequence-to-expression models on personal genome and transcriptome data},
    elocation-id = {2024.09.23.614632},
    year = {2024},
    doi = {10.1101/2024.09.23.614632},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2024/09/25/2024.09.23.614632},
    eprint = {https://www.biorxiv.org/content/early/2024/09/25/2024.09.23.614632.full.pdf},
    journal = {bioRxiv}
}
```
We fine-tune Enformer [1] in our work. We use the PyTorch port of Enformer available at https://github.com/lucidrains/enformer-pytorch as the base model. Our code is structured as follows:
- `finetuning/`: Scripts for fine-tuning Enformer on personal genome and transcriptome data using the various training strategies described in our paper.
- `fusion/`: Code to run variant-based baseline methods.
- `analysis/`: Scripts used for analysing predictions and generating figures.
- `process_geuvadis_data/`: Scripts for processing personal gene expression data from the GEUVADIS consortium. The processed data used in our experiments can be found at `process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz`.
- `process_sequence_data/`: Scripts to obtain personal genome sequences for individuals with matching gene expression data.
- `process_enformer_data/`: Code to build Enformer training data from Basenji2 training data by expanding input sequences. This data is used for joint training along with the personal genome and transcriptome data.
- `process_Malinois_MPRA_data/`: Code to download and format the MPRA data collected by Siraj et al. [2] from ENCODE. This data is also used for joint training.
- `vcf_utils/`: Miscellaneous utils used for processing VCF files.

The fine-tuned models used in our work are available at https://huggingface.co/anikethjr/finetuning-enformer/tree/main in the ``saved_models`` directory. We also provide the pre-processed personal genome and transcriptome data used in our experiments at https://huggingface.co/anikethjr/finetuning-enformer/tree/main/data.

# Steps to reproduce our main results:

We provide instructions to reproduce the main results of our work. The code is designed to run on a machine with multiple GPUs. We will update this README with further instructions to reproduce supplementary results in the future.

## Fine-tuning Enformer on personal genome and transcriptome data

1. Clone the repository:
   ```bash
   git clone https://github.com/ni-lab/finetuning-enformer.git
   cd finetuning-enformer
   ```

2. Create a Anaconda/Mamba environment with the required dependencies using the provided `env.yaml` file:
   ```bash
   conda env create -f env.yaml
   conda activate finetuning-enformer
   ```

3. Download the pre-processed personal genome and transcriptome data from the above link and place it in the `data/` directory. Use gzip to decompress the "*.h5.gz" files. The data should be structured as follows:
   ```
   data/
       train.h5
       val.h5
       test.h5
       rest_unseen_filtered.h5
   ```

4. First, we use single sample regression to fine-tune Enformer. Each of the following commands fine-tunes Enformer using a different seed. The results of these runs are saved in the `saved_models/` directory. The `--resume_from_checkpoint` flag is used to resume training from the last checkpoint if it exists, allowing for training to be continued from a previous run.
    ```bash
    NCCL_P2P_DISABLE=1 python finetuning/train_single_counts_parallel_h5_dataset.py data/train.h5 data/val.h5 single_regression_counts saved_models/ --batch_size 2 --lr 0.0001 --weight_decay 0.001 --use_scheduler --warmup_steps 1000 --data_seed 42 --resume_from_checkpoint
    NCCL_P2P_DISABLE=1 python finetuning/train_single_counts_parallel_h5_dataset.py data/train.h5 data/val.h5 single_regression_counts saved_models/ --batch_size 2 --lr 0.0001 --weight_decay 0.001 --use_scheduler --warmup_steps 1000 --data_seed 97 --resume_from_checkpoint
    NCCL_P2P_DISABLE=1 python finetuning/train_single_counts_parallel_h5_dataset.py data/train.h5 data/val.h5 single_regression_counts saved_models/ --batch_size 2 --lr 0.0001 --weight_decay 0.001 --use_scheduler --warmup_steps 1000 --data_seed 7 --resume_from_checkpoint
    ```

5. Next, we use pairwise regression to fine-tune Enformer.
    ```bash
    NCCL_P2P_DISABLE=1 python finetuning/train_pairwise_regression_parallel_h5_dataset.py data/train.h5 data/val.h5 regression saved_models/ --batch_size 1 --lr 0.0001 --weight_decay 0.001 --use_scheduler --warmup_steps 1000 --data_seed 42 --resume_from_checkpoint
    NCCL_P2P_DISABLE=1 python finetuning/train_pairwise_regression_parallel_h5_dataset.py data/train.h5 data/val.h5 regression saved_models/ --batch_size 1 --lr 0.0001 --weight_decay 0.001 --use_scheduler --warmup_steps 1000 --data_seed 97 --resume_from_checkpoint
    NCCL_P2P_DISABLE=1 python finetuning/train_pairwise_regression_parallel_h5_dataset.py data/train.h5 data/val.h5 regression saved_models/ --batch_size 1 --lr 0.0001 --weight_decay 0.001 --use_scheduler --warmup_steps 1000 --data_seed 7 --resume_from_checkpoint
    ```

6. Then we use pairwise classification to fine-tune Enformer.
    ```bash
    NCCL_P2P_DISABLE=1 python finetuning/train_pairwise_classification_parallel_h5_dataset_dynamic_sampling_dataset.py data/train.h5 data/val.h5 classification saved_models/ --batch_size 1 --lr 0.0001 --weight_decay 0.001 --data_seed 42 --resume_from_checkpoint
    NCCL_P2P_DISABLE=1 python finetuning/train_pairwise_classification_parallel_h5_dataset_dynamic_sampling_dataset.py data/train.h5 data/val.h5 classification saved_models/ --batch_size 1 --lr 0.0001 --weight_decay 0.001 --data_seed 97 --resume_from_checkpoint
    NCCL_P2P_DISABLE=1 python finetuning/train_pairwise_classification_parallel_h5_dataset_dynamic_sampling_dataset.py data/train.h5 data/val.h5 classification saved_models/ --batch_size 1 --lr 0.0001 --weight_decay 0.001 --data_seed 7 --resume_from_checkpoint
    ```

## References:
1. Avsec, Žiga, et al. "Effective gene expression prediction from sequence by integrating long-range interactions." Nature methods 18.10 (2021): 1196-1203.
2. Siraj, Layla, et al. "Functional dissection of complex and molecular trait variants at single nucleotide resolution." bioRxiv (2023).
