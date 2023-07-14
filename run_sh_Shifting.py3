#!/usr/bin/env python
import sys
import argparse
import os
import matplotlib

import pandas as pd
import numpy as np
import gseapy as gp
import seaborn as sns
import matplotlib.font_manager as fm

from scipy import stats
from pandas import DataFrame as df
from matplotlib import pyplot as plt
from matplotlib.ft2font import FT2Font
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def get_font(*args, **kwargs):
    return FT2Font(*args, **kwargs)

fm.get_font = get_font

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


'''
Global paths: do not change unless necessary
'''
poseidon_home = '/home/dyang-server/hdd/Yue/PROJECTS/POSEIDON/'
data_home = poseidon_home + 'data/'
gsea_home = poseidon_home + 'GSEA/'
cmap_home = data_home + 'CMAP2020_Cancer/'
panel_home = poseidon_home + 'panels/Figure2/'

def main(args):



    '''
    Read EC signatures
    '''
    gene_sets = {}
    with open(gsea_home + '/gene_sets/NREC_REC_91061.gmt', 'r') as f:
        for lines in f:
            lines = lines.rstrip().split('\t')
            gene_sets[lines[0]] = lines[2:]

    ec_color = df(index=gene_sets['NREC_profile'] + gene_sets['REC_profile'], columns=['colors'])
    ec_color.loc[gene_sets['NREC_profile'], 'colors'] = 'crimson'
    ec_color.loc[gene_sets['REC_profile'], 'colors'] = 'royalblue'

    # L1000 gene info
    bing_landmark = pd.read_csv(data_home + 'CMAP2020_Cancer/landmark_and_bings_L1000.csv',
                                header=0, index_col=0, sep=',', dtype={'Official NCBI gene symbol': 'str'}, converters={'Official NCBI gene symbol': None})
    '''
    Read input cancer signature info
    '''
    sig_info = pd.read_csv(data_home + 'CMAP2020_Cancer/' + args.cancer_path + '/' + args.cancer_type + '_2020_tash_sig_info.csv',
                           header=0, index_col=0, sep=',')
    cell_line = sig_info['cell_mfc_name'].unique()


    # shRNA perturbations that targeting these genes should be able to inhibit the expression of sig_defc_91061_up
    ## read original lv5 signature matrix
    trt_sh_original = pd.read_csv(data_home + 'CMAP2020_Cancer/' + args.cancer_path + '/CMAP2020_lv5_trt_sh_merged.csv',
                                       header=0, index_col=0, sep=',')
    trt_sh_original.index = trt_sh_original.index.astype(int)
    trt_sh_original = trt_sh_original[trt_sh_original.index.isin(bing_landmark.index)].rename(index=bing_landmark['Official NCBI gene symbol'])

    '''
    Read shift ability results
    '''
    shift_shRNA = pd.read_csv(poseidon_home + 'shift_ability/' + args.cancer_path + '_trt_sh.csv', header=0, index_col=0, sep=',')

    '''
    Visualization of shift ability for shRNAs with considerable KD efficiency: EC <= -1
    '''
    for cell in cell_line:

        print('running for ' + cell + '...')
        tmp_save = panel_home + 'direct_target_' + cell
        tmp_shift = shift_shRNA[shift_shRNA.index.isin(sig_info[sig_info['cell_mfc_name'] == cell].index)]

        # experiments directly targeting NREC and REC
        tar_dir_EC = tmp_shift[tmp_shift['cmap_name'].isin(gene_sets['NREC_profile'] + gene_sets['REC_profile'])].index

        # KO eff available
        ko_eff = trt_sh_original[trt_sh_original.index.isin(df(tmp_shift.loc[tar_dir_EC, 'cmap_name'])['cmap_name'])].index
        ko_eff_cal = df(index=tar_dir_EC, columns=['zexpr_lv5', 'gene'])
        for exp in tar_dir_EC:
            ko_eff_cal.at[exp, 'zexpr_lv5'] = trt_sh_original.loc[tmp_shift.loc[exp, 'cmap_name'], exp]
            ko_eff_cal.at[exp, 'gene'] = tmp_shift.loc[exp, 'cmap_name']

        # select the experiments with KO efficiency greater than 1.5 (z-score, decrease at 1.5 std)
        top_ko_eff = ko_eff_cal[ko_eff_cal['zexpr_lv5'] <= -1]
        if top_ko_eff.shape[0] == 0:
            print('KD efficiency fail to pass the criteria, skip the cell line ...')
            continue
        else:
            if 'direct_target_' + cell not in os.listdir(panel_home):
                os.mkdir(panel_home + 'direct_target_' + cell)

        top_ko_eff['NREC_enr'] = tmp_shift['NREC_profile']
        top_ko_eff['REC_enr'] = tmp_shift['REC_profile']
        top_ko_eff['shift_ability'] = tmp_shift['shift_ability']

        top_ko_eff_nrec = top_ko_eff[top_ko_eff['gene'].isin(gene_sets['NREC_profile'])]
        top_ko_eff_rec = top_ko_eff[top_ko_eff['gene'].isin(gene_sets['REC_profile'])]

        # add colors
        top_ko_eff.loc[top_ko_eff_nrec.index, 'EC_profile'] = 'crimson'
        top_ko_eff.loc[top_ko_eff_rec.index, 'EC_profile'] = 'royalblue'


        # visualize for shNRECs
        plt.figure(figsize=(3, 4))
        sns.violinplot(x='variable', y='value',
                       inner='quartile',
                       data=pd.melt(top_ko_eff.loc[top_ko_eff_nrec.index, ['NREC_enr', 'REC_enr']]),
                       palette={'NREC_enr': 'crimson', 'REC_enr': 'royalblue'})

        plt.scatter(np.zeros(len(top_ko_eff_nrec.index)), top_ko_eff.loc[top_ko_eff_nrec.index, 'NREC_enr'], s=5, c='k')
        plt.scatter(np.ones(len(top_ko_eff_nrec.index)), top_ko_eff.loc[top_ko_eff_nrec.index, 'REC_enr'], s=5, c='k')
        for i in top_ko_eff_nrec.index:
            plt.plot([0, 1], [top_ko_eff.loc[i, 'NREC_enr'], top_ko_eff.loc[i, 'REC_enr']], c='k', linewidth=0.2)

        plt.axhline(y=0., c='k', ls='--')
        plt.ylabel('enrichment', fontsize=14)
        plt.xlabel('')
        plt.yticks(rotation=90, fontsize=14)
        plt.xticks(fontsize=14)
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(tmp_save + '/NREC_targeted.png', transparent=True, dpi=300)
        fig.savefig(tmp_save + '/NREC_targeted.pdf', transparent=True)

        # visualize for shRECs
        plt.figure(figsize=(3, 4))
        sns.violinplot(x='variable', y='value',
                       inner='quartile',
                       data=pd.melt(top_ko_eff.loc[top_ko_eff_rec.index, ['NREC_enr', 'REC_enr']]),
                       palette={'NREC_enr': 'crimson', 'REC_enr': 'royalblue'})

        plt.scatter(np.zeros(len(top_ko_eff_rec.index)), top_ko_eff.loc[top_ko_eff_rec.index, 'NREC_enr'], s=5, c='k')
        plt.scatter(np.ones(len(top_ko_eff_rec.index)), top_ko_eff.loc[top_ko_eff_rec.index, 'REC_enr'], s=5, c='k')
        for i in top_ko_eff_rec.index:
            plt.plot([0, 1], [top_ko_eff.loc[i, 'NREC_enr'], top_ko_eff.loc[i, 'REC_enr']], c='k', linewidth=0.2)

        plt.axhline(y=0., c='k', ls='--')
        plt.ylabel('enrichment', fontsize=14)
        plt.xlabel('')
        plt.yticks(rotation=90, fontsize=14)
        plt.xticks(fontsize=14)
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(tmp_save + '/REC_targeted.png', transparent=True, dpi=300)
        fig.savefig(tmp_save + '/REC_targeted.pdf', transparent=True)

        # merged distribution
        shift_shRNA_tar_dir = tmp_shift[tmp_shift.index.isin(tar_dir_EC)]
        for g in shift_shRNA_tar_dir.index:
            shift_shRNA_tar_dir.at[g, 'EC_profile'] = ec_color.loc[shift_shRNA_tar_dir.loc[g, 'cmap_name'], 'colors']
        top_ko_shRNA_shift = shift_shRNA[shift_shRNA.index.isin(top_ko_eff.index)]

        plt.figure(figsize=(6, 4))
        # sns.kdeplot(x='shift_ability', data=tmp_shift,
        #             # cumulative=True,
        #             label='all shRNAs', fill=True, color='grey')
        # sns.rugplot(x='modified', data=shRNA_shift, color='grey')
        sns.kdeplot(x='shift_ability', data=tmp_shift[tmp_shift.index.isin(top_ko_eff_nrec.index)],
                    # cumulative=True,
                    label='shNREC, hi-ko', fill=True, color='crimson', ls='--')
        # sns.rugplot(x='modified', data=shRNA_shift[shRNA_shift.index.isin(top_ko_eff_nrec.index)], color='crimson')
        sns.kdeplot(x='shift_ability', data=tmp_shift[tmp_shift.index.isin(top_ko_eff_rec.index)],
                    # cumulative=True,
                    label='shREC, hi-ko', fill=True, color='royalblue', ls='--')
        # sns.rugplot(x='modified', data=shRNA_shift[shRNA_shift.index.isin(top_ko_eff_rec.index)], color='royalblue')
        plt.title(cell, fontsize=16)
        plt.xlabel('Shift ability', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.axvline(x=0, ls=':', c='k')
        plt.xlim(-1.2, 1.2)
        plt.legend()
        plt.yticks(rotation=0, fontsize=14)
        plt.xticks(fontsize=14)
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(panel_home + '/shift_ability/' + cell + '.png', transparent=True, dpi=300)
        fig.savefig(panel_home + '/shift_ability/' + cell + '.pdf', transparent=True)

        # merged boxplot
        top_ko_shRNA_shift.loc[top_ko_eff_nrec.index, 'EC_profile'] = 'shNREC'
        top_ko_shRNA_shift.loc[top_ko_eff_rec.index, 'EC_profile'] = 'shREC'

        plt.figure(figsize=(2, 4))
        sns.boxplot(x='EC_profile', y='shift_ability', data=top_ko_shRNA_shift,
                    palette={'shNREC': 'crimson', 'shREC': 'royalblue'},
                    linewidth=1)
        plt.xlabel('')
        plt.ylabel('Shift ability', fontsize=12)
        plt.yticks(rotation=90, fontsize=12)
        plt.axhline(y=0., c='k', ls=':')
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(panel_home + '/shift_ability/' + cell + '_boxplot.png', transparent=True, dpi=300)
        fig.savefig(panel_home + '/shift_ability/' + cell + '_boxplot.pdf', transparent=True)


if __name__ == '__main__':
    ### Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualizing the shift ability of shECs')
    parser.add_argument('--cancer_path', type=str, required=True, help='folder name of target cancer')
    parser.add_argument('--cancer_type', type=str, required=True, help='cancer type that needs to be analyzed')
    args = parser.parse_args()
    print(args)
    main(args)
