Pipeline to prepare TPM files consists of three steps.

1) Conversion of RPKM to TPM and removal of low expression genes
    - script: convert_to_tpm.ipynb
    - input: GD660.GeneQuantRPKM.txt.gz
    - output: tpm.csv.gz
2) Removal of batch effects (i.e. normalization) via PCA
    - script: normalize.ipynb
    - input: tpm.csv
    - output: tpm_pca.csv
3) Annotate genes with names and top eQTLs
    - script: ../annotate_expr_df.py
    - input: tpm_pca.csv
    - output: tpm_pca_annot.csv
