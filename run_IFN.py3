#!/usr/bin/env python
import sys
import argparse
import os

import pandas as pd
import numpy as np
import gseapy as gp

from scipy import stats
from pandas import DataFrame as df

'''
Global paths: do not change unless necessary
'''
poseidon_home = '/home/dyang-server/hdd/Yue/PROJECTS/POSEIDON/'
data_home = '/home/dyang-server/hdd/Yue/PROJECTS/POSEIDON/data/'
gsea_home = '/home/dyang-server/hdd/Yue/PROJECTS/POSEIDON/GSEA/'
cmap_home = data_home + 'CMAP2020_Cancer/'

'''
Main implemention
'''

def main(args):
    '''
    Main function of POSEIDON
    '''

    ## load perturbation matrix
    def load_pert_matrix(cancer_path, cancer_type, pert_type):
        # landmark genes
        bing_landmark = pd.read_csv(cmap_home + 'landmark_and_bings_L1000.csv',
                                    header=0, index_col=0, sep=',',
                                    dtype={'Official NCBI gene symbol': 'str'},
                                    converters={'Official NCBI gene symbol': None})
        # signature info of the given cancer type
        sig_info = pd.read_csv(cmap_home + cancer_path + '/' + cancer_type + '_2020_tash_sig_info.csv',
                               header=0, index_col=0, sep=',')
        # perturbation matirx of the given pert type
        pert_matrix = pd.read_csv(cmap_home + cancer_path + '/CMAP2020_lv5_' + pert_type + '_merged.csv',
                                  header=0, index_col=0, sep=',')
        pert_matrix.index = pert_matrix.index.astype(int)

        # rename the index to gene symbols
        pert_matrix = pert_matrix[pert_matrix.index.isin(bing_landmark.index)].rename(index=bing_landmark['Official NCBI gene symbol'])

        return pert_matrix, sig_info

    ## create rank list for perturbation matrix
    def pre_rank_initialize(cancer_path, pert_matrix, pert_type):
        # check if cancer_path exist
        if cancer_path not in os.listdir(gsea_home):
            os.mkdir(gsea_home + cancer_path)
        # check if pert_type exist
        if pert_type not in os.listdir(gsea_home + cancer_path):
            os.mkdir(gsea_home + cancer_path + '/' + pert_type)
        # create rank list
        for p in pert_matrix.columns:
            tmp = df(pert_matrix[p]).sort_values(by=p, ascending=False)
            p_rename = '__'.join(p.split(':'))
            tmp.to_csv(gsea_home + cancer_path + '/' + pert_type + '/' + p_rename + '.rnk', header=None, sep='\t')

        return


    def pre_rank_run(cancer_path, pert_type, pert_matrix, sig_info, permutate_time):
        # out put dir
        if 'enr_result_' + pert_type not in os.listdir(gsea_home + cancer_path):
            os.mkdir(gsea_home + cancer_path + '/enr_result_' + pert_type)

        prerank_nes = df(index=['HALLMARK_INTERFERON_ALPHA_RESPONSE',
                                'HALLMARK_INTERFERON_GAMMA_RESPONSE',
                                'HALLMARK_INFLAMMATORY_RESPONSE'], columns=pert_matrix.columns)
        prerank_fdr = df(index=['HALLMARK_INTERFERON_ALPHA_RESPONSE',
                                'HALLMARK_INTERFERON_GAMMA_RESPONSE',
                                'HALLMARK_INFLAMMATORY_RESPONSE'], columns=pert_matrix.columns)
        prerank_es = df(index=['HALLMARK_INTERFERON_ALPHA_RESPONSE',
                                'HALLMARK_INTERFERON_GAMMA_RESPONSE',
                                'HALLMARK_INFLAMMATORY_RESPONSE'], columns=pert_matrix.columns)

        for p in pert_matrix.columns:
            p_rename = '__'.join(p.split(':'))
            rnk = pd.read_csv(gsea_home + cancer_path + '/' + pert_type + '/' + p_rename + '.rnk', header=None, sep="\t")
            pre_res = gp.prerank(rnk=rnk, gene_sets=gsea_home + '/gene_sets/h_immune_res_sub.gmt',
                                 processes=1,
                                 permutation_num=permutate_time, # reduce number to speed up testing
                                 outdir=None, format='png', seed=0, min_size=0, max_size=10000)
            prerank_nes[p] = pre_res.res2d['nes']
            prerank_fdr[p] = pre_res.res2d['fdr']
            prerank_es[p] = pre_res.res2d['es']

        prerank_fdr = prerank_fdr.T
        prerank_es = prerank_es.T
        prerank_nes = prerank_nes.T

        prerank_fdr['cmap_name'] = sig_info['cmap_name']
        prerank_fdr['nearest_dose'] = sig_info['nearest_dose']
        prerank_fdr['pert_idose'] = sig_info['pert_idose']

        prerank_es['cmap_name'] = sig_info['cmap_name']
        prerank_es['nearest_dose'] = sig_info['nearest_dose']
        prerank_es['pert_idose'] = sig_info['pert_idose']

        prerank_nes['cmap_name'] = sig_info['cmap_name']
        prerank_nes['nearest_dose'] = sig_info['nearest_dose']
        prerank_nes['pert_idose'] = sig_info['pert_idose']

        prerank_es.to_csv(gsea_home + cancer_path + '/enr_result_' + pert_type + '/es_hallmark_IFN.csv', sep=',')
        prerank_nes.to_csv(gsea_home + cancer_path + '/enr_result_' + pert_type + '/nes_hallmark_IFN.csv', sep=',')
        prerank_fdr.to_csv(gsea_home + cancer_path + '/enr_result_' + pert_type + '/fdr_hallmark_IFN.csv', sep=',')

        return prerank_es

    # load perturbation matrix
    print('Loading perturbation matrix for ' + args.cancer_type + ', ' + args.pert_type + '...')
    pert_matrix, sig_info = load_pert_matrix(cancer_path=args.cancer_path,
                                             cancer_type=args.cancer_type,
                                             pert_type=args.pert_type)
    print(args.cancer_type + ', ' + args.pert_type + ' dimension:')
    print(pert_matrix.shape)

    # Pre Rank initialization: skip if rank list created
    if args.pre_rank == 'Y':
        print('Creating pre-rank lists...')
        pre_rank_initialize(cancer_path=args.cancer_path, pert_matrix=pert_matrix, pert_type=args.pert_type)
        print('Pre-rank lists created')
    elif args.pre_rank == 'N':
        print('Pre-rank lists have already been created, skip...')
    else:
        print('pre_rank must be specified with Y or N...')
        sys.exit(0)

    # run pre-rank analysis
    prerank_es = pre_rank_run(cancer_path=args.cancer_path,
                              pert_type=args.pert_type,
                              pert_matrix=pert_matrix,
                              sig_info=sig_info,
                              permutate_time=args.permutate_time)


if __name__ == '__main__':
    ### Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculating enrichment of IFNs for input perturbation matrix')
    parser.add_argument('--cancer_path', type=str, required=True, help='folder name of target cancer')
    parser.add_argument('--cancer_type', type=str, required=True, help='cancer type that needs to be analyzed')
    parser.add_argument('--pert_type', type=str, required=True, help='perturbation type that needs to be analyzed')
    parser.add_argument('--pre_rank', type=str, required=True, help='whether to create the rank list, skip if this step has already been done')
    parser.add_argument('--permutate_time', type=int, required=True, help='number of permutation included in enrichment analysis')

    args = parser.parse_args()
    print(args)
    main(args)
