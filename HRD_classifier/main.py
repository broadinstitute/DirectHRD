#!/usr/bin/env python
# coding: utf-8

import argparse
import sigProfilerPlotting as sigPlt
from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as datadump
from SigProfilerExtractor import subroutines as sub
import pandas as pd
import SigProfilerAssignment as spa
from SigProfilerAssignment import decomposition as decomp
from SigProfilerAssignment import Analyzer as Analyze
import numpy as np
import glob
import pickle
import pkg_resources
import os

from .__version__ import __version__

def print_version():
    print(f"DirectHRD version: {__version__}")

def _EM(dat, sigs, feat, indx, maxitr=100, verbose=False):
    #print("#signatures used:", len(sigs))
    id_probs = []
    for sig in sigs:
        sig = sig.loc[indx, ]
        sig_prob = sig.filter(regex=feat, axis="index")

        sig_prob = sig_prob/sig_prob.sum()
        id_probs.append(sig_prob)

    id_probs_array = np.array(id_probs)
    
    probs = id_probs_array.copy()
    pis = np.ones(len(probs))/len(probs)

    converged = False
    nindels = int(dat.sum())
    newprobs = probs.copy()
    newpis = np.zeros(len(probs))
    
    alpha=1/nindels
    for i in range(maxitr):
        for idx in range(len(pis)):
            hidtable = pis[idx]*probs[idx]/np.sum(np.column_stack(probs)*pis, axis=1)
            hidtable = np.nan_to_num(hidtable)
            #dat = dat.astype('float')
            tmpSigWt = np.dot(hidtable, dat)
            newprobs[idx] = probs[idx]* (1-alpha) +  alpha*hidtable*dat/ tmpSigWt
            newpis[idx] = tmpSigWt/dat.sum()        
        dis = np.linalg.norm(newpis - pis)
        if dis < 1e-6:
            converged = True
            if verbose:
                print(f"converaged at {i} step")
            break
        pis = newpis.copy()
        probs = newprobs.copy()
        
    return pis, probs, converged


def mmm_classifier(inputmat, id8 = True, 
                   feat= ":Del:M", count_feat='5:Del:M', maxitr = 500, verbose=False):
        
    path = spa.__path__[0]
    cosmic_version=3.2
    ID83 = pd.read_csv(path+"/data/Reference_Signatures/GRCh37/COSMIC_v"+str(cosmic_version)+"_ID_GRCh37.txt", sep="\t", index_col=0)
    with open(pkg_resources.resource_filename('HRD_classifier', 'data/ID83_model.hrdneg.pickle'), 'rb') as handle:
        pcawg_neg_id = pickle.load(handle)
    neg_prob = pcawg_neg_id['ID'].apply(sum, axis=1)
    neg_prob = neg_prob / neg_prob.sum()
    
    with open(pkg_resources.resource_filename('HRD_classifier', 'data/ID83_model.hrdpos.pickle'), 'rb') as handle:
        pcawg_pos_id = pickle.load(handle)
    pos_prob = pcawg_pos_id['ID'].apply(sum, axis=1)
    pos_prob = pos_prob / pos_prob.sum()
        
    
    ##make sure we have enough count from each type
    indx = pcawg_neg_id['ID'].apply(sum, axis=1) > 10
    

    scores = []
    n_informative_del = []
    id6pis = []
    negpis = []
    considered = []
    dellen2 = []
    totdel = []
    mhdels = []
    del5_m2 = []
    samples = inputmat.columns.tolist()
    for sam in samples:
        sigs = [ID83['ID6'], neg_prob]
        if id8:
            sigs.append(ID83['ID8'])
        dat=inputmat[sam].loc[indx,].filter(regex=feat, axis='index').to_numpy(dtype='float')
        nindels = int(dat.sum())
        considered.append(nindels)
        nmut = inputmat[sam].loc[indx,].filter(regex=count_feat, axis='index').sum()
        m = inputmat[sam].loc[indx,].filter(regex="Del:M", axis='index').sum()
        del52 = inputmat[sam].loc[indx,].filter(regex="5:Del:M:[2-5]", axis='index').sum()
        del5_m2.append(del52)
        mhdels.append(m)
        dellen2.append(inputmat[sam].filter(regex='[2-5]:Del:').sum())
        totdel.append(inputmat[sam].filter(regex=':Del:').sum())
        n_informative_del.append(nmut)
        
        if dat.sum() == 0:
            scores.append(np.nan)
            id6pis.append(np.nan)
            negpis.append(np.nan)
            continue
        pis, probs, converged = _EM(dat, sigs, feat, indx, maxitr=maxitr, verbose=verbose)
        if id8:
            indel_scores = probs[0] * pis[0] / (probs[0] * pis[0] + probs[1] * pis[1] + probs[2]*pis[2])
        else:
            indel_scores = probs[0] * pis[0] / (probs[0] * pis[0] + probs[1] * pis[1])
        indel_scores = np.nan_to_num(indel_scores)
        indel_scores = pd.Series(indel_scores, index=sigs[0].loc[indx, ].filter(regex=feat, axis="index").keys())

        hrdscore = sum(inputmat[sam].loc[indx,].filter(regex=count_feat, axis='index') * indel_scores.filter(regex=count_feat, axis='index')) * (pis[0])
        hrdscore = round(hrdscore, 2)
        id6pis.append(pis[0])
        negpis.append(pis[1])
        if verbose:
            print(sam, hrdscore, pis[0], sep='\t')
        scores.append(hrdscore)
    result = pd.DataFrame({'sample':samples,
                           'HRDscore':scores,
                           'ID6_prob':id6pis,
                           'HRDneg_prob':negpis,
                           'n_informative_del':n_informative_del,
                           'mhdels': mhdels,
                           '5del_m2': del5_m2,
                           'considered': considered,
                           'del_2bp+':dellen2,
                           'totl_del': totdel})
    result['HRDscore'] = np.where(result['considered'] == 0, 1e-5, result['HRDscore'])
    result['frac_signal']=result['n_informative_del'].div(result['considered'])
    return result


## This exmaple uses GRch37 or HG19 reference
def directhrd_run(project, refgen, myfeat='Del:M|5:Del:R:0'):
    try:
        mutation_profile=datadump.SigProfilerMatrixGeneratorFunc("directhrd", refgen, project, exome=False,  bed_file=None, chrom_based=False, plot=False, gs=False)
    except Exception as e:
        print(f"An error occured: {e}")

    directhrd_res = mmm_classifier(mutation_profile['ID'], feat=myfeat, id8=True)
    return (directhrd_res)


#print mhdels
def print_mhdel(vdir, sam):
    alldf = pd.DataFrame()
    for f in glob.glob(f"{vdir}/output/vcf_files/ID/*.txt"):
        df = pd.read_csv(f, names=['sample', 'chrom', 'pos', 'class', 'ref', 'alt', 'code'], sep='\t')
        alldf = pd.concat([alldf, df])
    alldf['chrom'] = alldf['chrom'].astype('str')
    idx = alldf[alldf['sample']==sam]['class'].str.contains('Del:M')
    return (alldf[alldf['sample']==sam][idx].sort_values(by='chrom'))


def main():
    parser = argparse.ArgumentParser(prog="hrd_classifier", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_folder", type=str, help="input folder")
    parser.add_argument("--ref_version", "-r", required=True, type=str, choices=['GRCh37', 'GRCh38'], help="human referernce version")
    parser.add_argument("--output", "-o", default="directhrd.results.txt", type=str, help="result file")
    parser.add_argument("-s", "--sample", default="", type=str, help="print mhDels for a specific sample")
    parser.add_argument("-p", "--plot", default="", type=str, help="output directory for ID83 signature plot")
    args = parser.parse_args()
    if args.sample:
        print(print_mhdel(args.input_folder, args.sample))
        return
    if args.plot:
        sigPlt.plotID(os.path.join(args.input_folder, "output", "ID/directhrd.ID83.all"), args.plot, "directhrd", "83", percentage=True)
        return
    # Your code here
    print("HRD Classifier is running.")
    print_version()
    ### input folder contains only indel VCF files
    results = directhrd_run(args.input_folder, args.ref_version)
    print(f"Result was written to {args.output}.")
    results.to_csv(args.output, sep="\t", index=False)
    # You can place the code from your Jupyter notebook here or call other functions

if __name__ == "__main__":
    main()
