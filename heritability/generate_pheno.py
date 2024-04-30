import pandas as pd
df = pd.read_csv("/clusterfs/nilah/rkchung/data/finetuning-enformer/process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
gene_names = list(pd.read_csv("/clusterfs/nilah/rkchung/data/finetuning-enformer/finetuning/data/h5_bins_384_chrom_split/gene_class.csv").gene)

region_size = 49152
for g in gene_names:
    sub = df[df["our_gene_name"] == g]
    c = list(sub["Chr"])[0]
    s = list(sub["Coord"])[0]-int(region_size/2)
    e = list(sub["Coord"])[0]+int(region_size/2)
    sub = sub[sub.columns[4:-9]].T.reset_index()
    sub["index2"] = sub["index"]
    sub = sub[["index2"]+list(sub.columns[:2])]
    sub.to_csv(f"/clusterfs/nilah/rkchung/data/geuvadis/geuvadis_pheno/{g}_{c}_{s}_{e}.pheno", sep="\t", index=False, header=False)

