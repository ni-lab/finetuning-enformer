import h5py
import pandas as pd

# Loads ISM scores from Enformer for each variant given dataframe of variant dosages
# Returns variant dosage dataframe * ISM scores
def _load_sar_scores_for_chrom(
    chrom: str,
    variants: pd.DataFrame,
    track_idxs: list,
    track_weights: list,
    enformer_predictions_dir: str = "/clusterfs/nilah/rkchung/data/basenji/enformerism",
):
    h5_path = os.path.join(
        enformer_predictions_dir, f"1000G.MAF_threshold=0.005.{chrom}.h5"
    )

    # Read in enformer sar scores and positions
    with h5py.File(h5_path, "r") as f:
        chr = f["chr"][:].astype(str)
        pos = f["pos"][:].astype(str)
        ref = f["ref"][:].astype(str)
        alt = f["alt"][:].astype(str)
        scores = np.abs(f["SAR"][:, track_idxs].astype(float))
        weighted_scores = scores @ np.asarray(track_weights)
    sar_df = pd.DataFrame(np.array([chr, pos, ref, alt, weighted_scores]).T, 
                          columns=["chr", "pos", "ref", "alt", "score"])
    sar_df["fullpos"] = sar_df[["chr", "pos"]].apply(
        lambda cols: ":".join([str(c) for c in cols]), axis=1)

    # Merge ISM scores with genotype dataframe
    variants = variants.set_index("fullpos").join(sar_df.set_index("fullpos"), rsuffix="_x", lsuffix="_y")
    variants["score"] = variants["score"].astype(float)
    mean_ism = variants["score"].mean()

    # Flip genotypes if ref and alt are flipped in ISM
    variants["flip"] = variants[["ref_x", "ref_y", "alt_x", "alt_y"]].apply(
        lambda a: ((a[0]==a[3]) and (a[1]==a[2])), axis=1)
    variants["score"] = variants[["ref_x", "ref_y", "alt_x", "alt_y", "score"]].apply(
        lambda a: a[4] if (((a[0]==a[1]) and (a[2]==a[3])) or ((a[0]==a[3]) and (a[1]==a[2]))) else np.nan, axis=1)
    
    # Fill in scores that were missing with mean value
    variants["score"] = variants["score"].fillna(mean_ism)
    
    # Remove variants if there are missing values (positions at beginnning and end of chrom)
    variants = variants.loc[~(variants==-2).any(1)]
    indivs = [c for c in variants.columns if ("HG" in c) or ("NA" in c)]
    
    # Flip genotypes that don't match
    variants[indivs] = variants[indivs]\
        .multiply((-2*variants["flip"].astype(int)+1), axis="index")\
        .add(2*variants["flip"].astype(int), axis="index")
    print(variants)

    # Multiply genotypes by the ISM weights
    variants[indivs] = variants[indivs].multiply(variants["score"], axis="index")
    variants[indivs] = (variants[indivs].T-variants[indivs].mean(1)).T
    return variants

# Constructs output filename suffix
def construct_filename(run, only_lcl, compare_enformer, short_region, subset_yri, predixcan_residual):
    suff = ""
    suff += "predixcan" if ("predixcan" in run) else ""
    suff += "ism" if ("ism" in run) else ""
    suff += "enformer" if ("enformer" in run) else ""
    suff += ".weights"
    suff += ".lcl" if only_lcl else ""
    suff += ".197KB" if compare_enformer else (".13KB" if short_region else "")
    suff += f".{run}" if (("predixcan" not in run) and ("enformer" not in run)) else ""
    suff += ".alltracks" if "enformer" in run else ""
    suff += ".nocommon" if "nocommon" in run else ""
    suff += ".yri.logtpm" if subset_yri else ""
    suff += ".residual" if predixcan_residual else ""
    return suff

# Load location of TSS for each gene
def get_gene_df(genes_file):
    gene_df = pd.read_csv(genes_file, names=["geneId", "chr", "tss", "name", "strand"])
    gene_df["name"] = gene_df.apply(lambda x: x["name"] if not any(x.isnull()) else x["geneId"], axis=1).str.lower()
    gene_df = gene_df.sort_values(by="name")
    return gene_df
