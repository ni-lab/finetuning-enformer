import allel
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from operator import methodcaller

from scipy.stats import spearmanr

from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
#import glmnet_python
#import glmnet

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_predict
from skopt import BayesSearchCV

from sklearn.linear_model import _coordinate_descent

from sklearn.exceptions import ConvergenceWarning
import warnings

from itertools import islice
from joblib import Parallel, delayed

import sys
import h5py
from helpers import *

data_dir = "/clusterfs/nilah/rkchung/data/geuvadis/"
ruchir_dir = "/clusterfs/nilah/rkchung/data/finetuning-enformer/"

warnings.filterwarnings("ignore", category=ConvergenceWarning)
def fit_clf(X, expr, tr_indiv=None, t_indiv=None, center=False):
    all_indiv = tr_indiv if (t_indiv is None) else tr_indiv+t_indiv
    if center:
        expr[all_indiv] = ((expr[all_indiv] - expr[all_indiv].mean()) / expr[all_indiv].std()).replace([np.inf, -np.inf], 0).fillna(0)

    exprarr = expr[tr_indiv].to_numpy().T
    alphas = _coordinate_descent._alpha_grid(X[tr_indiv].to_numpy().T, 
            exprarr.reshape(-1,1), l1_ratio=0.5)
    param_grid = {'alpha': alphas,
                  'l1_ratio': [0.5]}
    cv = 10

    if t_indiv is not None:
        best_model, best_alpha, y_pred, score, coef_ = \
                cv_train(np.transpose(X[tr_indiv].to_numpy()), 
                                  exprarr, param_grid, 
                                  cv, "r2")
        y_pred_t = best_model.predict(np.transpose(X[t_indiv].to_numpy()))
        t_r2 = best_model.score(np.transpose(X[t_indiv].to_numpy()), np.transpose(expr[t_indiv]))
        t_spearr = spearmanr(expr[t_indiv].to_numpy().T, list(y_pred_t))[0]
        spearr = spearmanr(exprarr, list(y_pred))[0]
        return score, t_r2, spearr, t_spearr, best_alpha, list(y_pred) + list(y_pred_t), coef_
    else:
        # Train the model and get the best model and cross-validated predictions
        best_model, best_alpha, y_pred, score, coef_ = \
                cv_train(np.transpose(X[tr_indiv].to_numpy()), 
                                  exprarr, param_grid, 
                                  cv, "r2")
        spearr = spearmanr(exprarr, list(y_pred))[0]
        return score, spearr, best_alpha, list(y_pred), coef_

def cv_train(X, y, param_grid, cv, scoring, use_bayessearch=False):
    # Define the grid search object
    if use_bayessearch:
        model = ElasticNet(l1_ratio=0.5, max_iter=500)
        cvmodel = BayesSearchCV(model, param_grid, n_iter=500, cv=cv, scoring=scoring)
    else:
        cvmodel = ElasticNetCV(l1_ratio=0.5, max_iter=500, cv=cv, n_jobs=1)

    # Fit the grid search object to the data
    cvmodel.fit(X, y)

    # Get the best hyperparameters and model
    if use_bayessearch:
        best_alpha = cvmodel.best_params_['alpha']
        cvmodel = cvmodel.best_estimator_
    else:
        best_alpha = cvmodel.alpha_

    # Define Elastic Net model for cv training predictions
    model = ElasticNet(alpha=best_alpha, l1_ratio=0.5, max_iter=500)

    # Get cross-validated predictions using the best model
    y_pred = cross_val_predict(model, X, y, cv=cv)

    # Compute the scoring metric of the predictions
    score = r2_score(y, y_pred)

    return cvmodel, best_alpha, y_pred, score, cvmodel.coef_

def train_model(gt, row, covar, indiv, tr_indiv, reg_out=False, t_indiv=None, t_young_indiv=None, run="predixcan", subset_yri=False, region_size=1e6, center=False, only_lcl=False):
    num_out = 5 if t_indiv is None else 7 
    X = None
    # Load enformer intermediate activations (feel free to delete)
    if (run!="predixcan") and (run!="predixcanism"):
        genes_file = "/global/home/users/rkchung/personalized-expression-benchmark/data/gene_list.csv"
        gene_df = get_gene_df(genes_file)
        matching = gene_df[gene_df["geneId"]==row["gene_id"]]
        if len(matching)==0:
            print("Gene missing TSS information: %s" % row["gene_id"])
            return [row["gene_id"]]+([0] * num_out)
        genename = matching.iloc[0]["name"]
        preds_file_name = f"/clusterfs/nilah/rkchung/data/personalexpr/enformer%sout%s/{genename}/{genename}.h5" % ("" if "enformer" in run else "act", "_nocommon" if "nocommon" in run else "")
        if not os.path.exists(preds_file_name):
            print("missing predictions for " + row["gene_id"])
            return [row["gene_id"]]+([0] * num_out)
        f = h5py.File(preds_file_name, "r") 
        with open("/clusterfs/nilah/connie/enformer/data/same_length_inds.txt", "r") as fn: 
            samples = fn.read().split("\n")[:-1]
        key = "preds"
        if (run is not None) and ("enformer" not in run):
            key = run
        sizes = {"preds":5, "transformer":3, "conv_tower.5":5, "conv_tower.3":17, 
                "conv_tower.1":65, "stem": 257}
        if "nocommon" in run:
            sizes["preds"] = 10
        X = np.array(f[key]).reshape(len(samples), 2, -1, sizes[key]).mean((1,3)).T
        if only_lcl:
            X = X[[5110],:]
        X = pd.DataFrame(X, columns=[samples])
        X = np.log10(X+0.001)
        variants = None

    # Load predixcan variants for single gene
    if "predixcan" in run:
        # pandas df of genotypes (indiv x positions)
        gene_gt = gt.loc[(gt["Chrom"]==row["#chr"]) & 
                         (gt["Pos"]>=(row["start"]-int(region_size/2))) & 
                         (gt["Pos"]<=(row["start"]+int(region_size/2)))] 
        variants = gene_gt[["Chrom", "Pos"]].apply(lambda v: f"{v[0]}:{v[1]}", axis=1)
        gene_gt = gene_gt[gt.columns.difference(["Pos", "Chrom"])]
        if gene_gt.shape[0] == 0:
            print("Zero variants for gene %s" % row["gene_id"])
            return [row["gene_id"]]+([0] * num_out)
        if X is not None:
            X.columns = X.columns.get_level_values(0)
            empty_columns_X = X[samples].columns[X[samples].isnull().all()]
            X = pd.concat([X[samples].apply(pd.to_numeric).astype(float), gene_gt[samples].apply(pd.to_numeric).astype(float)], ignore_index = True)
        else:
            X = gene_gt
    X = (X.apply(lambda x: (x - x.mean())/x.std(), axis=1)).fillna(0).replace([np.inf, -np.inf], 0) # normalize all predictors

    if reg_out: # Regress the covariates out from expression data
        skclf = LinearRegression()
        skclf.fit(np.transpose(covar[indiv].to_numpy()), np.transpose(row[indiv].to_numpy()))
        row[indiv] = row[indiv] - skclf.predict(np.transpose(covar[indiv].to_numpy()))
    print("Fitting %s" % row["gene_id"])    
    # Fit single gene model
    outs = fit_clf(X, row, tr_indiv, t_indiv, center=center)
    return [row["gene_id"]] + list(outs) 

if __name__=="__main__":
    # Commandline arguments for multiple jobs: 
    #    chromosome (recommended) e.g. 1-22
    #    chunk of chromosome (optional) e.g. 1-10
    # Run like: python run_geuvadis.py {chromosome} {chunk}
    if len(sys.argv) > 1:
        run_chrom = sys.argv[1]
        print("Chrom %s" % run_chrom)
        if len(sys.argv) > 2:
            chunks = 10
            chunk = sys.argv[2]
            print("Chunk %s" % chunk)
    subset_yri = True # train on non-yri, test on yri indviduals, expression dataset from ruchir (TPM)
    short_region = False # 13kb context
    compare_enformer = False # 197kb context
    subset_eqtl_genes = False # 62 genes
    predixcan_residual = False # Train model on top of predixcan residual expression values
    only_lcl = False # Only use enformer track corresponding to the LCL CAGE predictions
    runs = ["predixcan"] # Type of model to run 
    # possible run names: "enformer", "predixcan", "predixcanenformer", 
    #                     "enformernocommon", "transformer", "conv_tower.5", 
    #                     "conv_tower.3", "conv_tower.1", "stem"
    center = False

    split_tt = False # Split individuals into train and validation sets
    reg_out = False # Regress out covariates from gene expression (e.g. PEER)

    for run in runs:
        if short_region:
            region_size = 12800
        elif compare_enformer:
            region_size = 197e3
        else:
            region_size = 1e6
        
        # Construct out filename
        suff = construct_filename(run, only_lcl, compare_enformer, short_region, subset_yri, predixcan_residual)

        if 'run_chrom' in locals() and run_chrom!="":
            outfile = data_dir+"results/results.geuvadis.norm.%s.chr%s" % (suff, run_chrom)
            if 'chunk' in locals() and chunk != "":
                outfile = data_dir+"results/results.geuvadis.norm.%s.chr%s_%s" % (suff, run_chrom, chunk)
        else:
            outfile = data_dir+"results/results.geuvadis.norm.%s" % suff
        outfile += ".2"
        print(f"Name of outfile: {outfile}")

        if "predixcan" in run:
            # Load predixcan genotype data
            if 'run_chrom' in locals() and run_chrom!="":
                # TODO: change file to original GEUVADIS data (separated by chromosome) and filter for maf/hwe on on the fly
                callset = allel.read_vcf(data_dir+"GEUVADIS.mafhwe.chr%s.genotypes.vcf" % run_chrom)
            else:
                callset = allel.read_vcf(data_dir+"GEUVADIS.allchroms.mafhwe.snps.vcf")

            gt = np.sum(callset["calldata/GT"], axis=-1).astype(np.int8)
            chrom = callset['variants/CHROM']
            pos = callset["variants/POS"]
            indiv = [name.split("_")[1] for name in callset['samples']]
            print(f"Shape of genotype matrix: {gt.shape}")
            gt = pd.DataFrame(data=gt, columns=indiv)
            pos = pd.DataFrame(data=pos, columns=["Pos"])
            chrom = pd.DataFrame(data=chrom, columns=["Chrom"])
            gt = pd.concat((chrom, pos, gt), axis=1)
            gt["Chrom"] = gt["Chrom"].astype(str)
        else:
            # Load geuvadis individuals for enformer weight model 
            with open("/clusterfs/nilah/connie/enformer/data/same_length_inds.txt", "r") as f: 
                indiv = f.read().split("\n")[:-1]
            gt = None

        if "ism" in run:
            # Pull ism scores for all variants (model uses ism scores instead of genotype dosage)
            gt["ref"] = callset["variants/REF"]
            gt["alt"] = [a[0] for a in callset["variants/ALT"]]
            gt["fullpos"] = gt[["Chrom", "Pos"]].apply(lambda cols: "chr"+":".join([str(c) for c in cols]), axis=1)
            gt = _load_sar_scores_for_chrom(run_chrom, gt, [69, 5110], [0.5,0.5])
            gt = gt.reset_index().drop(
                    ["fullpos", "ref_x", "ref_y", "alt_x", "alt_y", 
                    "score", "pos", "chr", "flip"], axis=1)

        # Load expression data
        if predixcan_residual:
            expr = pd.read_csv(data_dir+\
                    "predixcan.residual.tpm_pca_annot.csv", index_col=0)
            expr["Chr"] = expr["Chr"].astype(str)
        elif subset_yri:
            expr = pd.read_csv(ruchir_dir+\
                    "process_geuvadis_data/log_tpm/corrected_log_tpm.annot.csv.gz")
            expr = expr[expr.columns[:-6]]
        else:
            expr = pd.read_csv(data_dir+\
                    "GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz", 
                    sep="\t")

        # Reformat columns of expression dataframe
        expr = expr.rename({"Chr": "#chr", "Coord": "start"}, axis=1)
        expr["gene_id"] = expr["TargetID"].apply(lambda g: g.split(".")[0])
        expr["end"] = expr["start"]+1
        
        # Run analysis on one partition of the genes
        if 'run_chrom' in locals() and run_chrom!="":
            expr = expr[expr["#chr"]==run_chrom]
            if 'chunk' in locals() and chunk!="":
                expr = np.array_split(expr, chunks)[int(chunk)]

        indiv = list(set(expr.columns).intersection(set(indiv)))
        indiv = list(pd.read_csv("/clusterfs/nilah/rkchung/data/expred/geuvadis.phased.samples.txt")["indiv"])

        # Split individuals into a train and validation split 
        indiv = np.array(indiv)
        t_indiv = None
        if subset_yri:
            indivs = pd.read_csv(ruchir_dir+"finetuning/data/bins_100/splits_for_predixcan/samples.csv")
            tr_indiv = list(indivs[indivs["population"]=="Non-YRI"]["sample"])
            t_indiv = list(indivs[indivs["population"]=="YRI"]["sample"])
        else:
            if split_tt:
                split = 0.2 
                perm = np.random.permutation(len(indiv))
                t_indiv = indiv[perm[:int(split*len(indiv))]]
                tr_indiv = indiv[perm[int(split*len(indiv)):]]
            else:
                perm = np.random.permutation(len(indiv))
                tr_indiv = indiv[perm]

        # Run on a predefined subset of genes
        if subset_eqtl_genes:
            geneset = np.load(ruchir_dir+"finetuning/data/bins_100/splits_for_predixcan/yri_test_genes.npy", allow_pickle=True)
            expr = expr[expr["our_gene_name"].apply(lambda g: g in geneset)]
        else:
            eval_genes = pd.read_csv("/global/home/users/rkchung/evaluation_genes3000.txt", 
                                     names=["gene_id", "Chr", "Position", "Common", "Strand"])
            geneset = list(eval_genes.gene_id)
            expr = expr[expr["gene_id"].apply(lambda g: g in geneset)]

        covars = pd.read_csv(data_dir+"GD462.peerfactors.csv")
        covars = covars[["V%s" % str(v+1) for v in range(15)]]
        if not predixcan_residual:
            if subset_yri:
                covars.index =  expr.columns[4:-5]
                covars = covars.T
            else:
                covars.index =  expr.columns[4:-2]
                covars = covars.T

        # Parallelize training process
        result_items = Parallel(n_jobs=56, backend='threading')(delayed(
                train_model)(gt, row, covars, indiv, tr_indiv, reg_out=reg_out,
                    t_indiv=t_indiv, run=run, subset_yri=subset_yri, 
                    region_size=region_size, center=center, only_lcl=only_lcl) 
                for g, row in tqdm(islice(expr.iterrows(), 0, None), 
                    desc="Training Genes", total=(len(expr))))

        res_columns = ["gene_id", "cv_r2"]
        if t_indiv is not None:
            res_columns += ["t_r2", "cv_spearmanr"]
        res_columns += ["spearmanr", "best_alpha", "preds", "weights"]
        result_items = pd.DataFrame(np.array(result_items).T, res_columns)
        result_items.to_pickle(f"{outfile}.df")
