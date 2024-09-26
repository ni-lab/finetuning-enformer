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
- `process_Malinois_MPRA_data/`: Code to download and format the MPRA data collected by Siraj et al. [2] from ENCODE.
- `vcf_utils/`: Miscellaneous utils used for processing VCF files.

## References:
1. Avsec, Žiga, et al. "Effective gene expression prediction from sequence by integrating long-range interactions." Nature methods 18.10 (2021): 1196-1203.
2. Siraj, Layla, et al. "Functional dissection of complex and molecular trait variants at single nucleotide resolution." bioRxiv (2023).
