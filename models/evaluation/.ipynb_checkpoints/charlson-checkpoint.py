# +-------------------------------------------------------------------------------------------------+
# | charlson.py: functions to analyze and plot comorbidity-vs-risk curves                           |
# |                                                                                                 |
# | Eliane Röösli (2020)                                                                            |
# +-------------------------------------------------------------------------------------------------+


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import statsmodels.api as sm
from matplotlib.lines import Line2D
from benchmarks.common_tools import find_map_key
from models.evaluation.configs import groups_excluded, labeldict, colors, is_public_map, paths, charlson_curves


# ---------------------------- FINAL GROUPED PLOT -------------------------------

def charlson_grouped_all(listfiles):
        
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    lowess = sm.nonparametric.lowess
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in charlson_curves]
    quantiles = np.arange(0,100,1)
    ncol = 1
    
    for idx_var, dem_var in enumerate(['Insurance', 'Gender', 'Ethnicity']):
        groups = listfiles[0][dem_var].unique()
        labels = []
        for idx, listfile in enumerate(listfiles):
            ctr = 0
            for group in groups:
                name = find_map_key(dem_var, group)
                if name not in groups_excluded:
                    charlson = [listfile[(listfile['prediction']*100>=quantile) & (listfile[dem_var]==group)]['comorb2'].mean() for quantile in quantiles]
                    axs[idx_var, idx].scatter(x=quantiles, y=charlson, s=15, marker='x', c=charlson_curves[ctr], alpha=0.7)
                    z = lowess(endog=charlson, exog=quantiles, frac=0.3, it=0, return_sorted=True)
                    axs[idx_var, idx].plot(z[:,0], z[:,1], c=charlson_curves[ctr])
                    ctr += 1
                    if idx == 0:
                        labels.append(labeldict[name])
            if idx_var == 2:
                ncol = 2
            axs[idx_var, idx].legend(lines, labels, loc='upper left',fontsize=12, ncol=ncol)
        axs[idx_var,1].tick_params(left=False)

    for ax in axs.flat:
        ax.grid(c=colors['grid'], zorder=100)
        ax.tick_params(labelsize=12)
        ax.set_axisbelow(True)
        ax.set_xlim(-3,103)
        ax.set_ylim(-0.5,13)
        ax.set_yticks([2,5,8,11])
        plt.setp(ax.spines.values(), color=colors['frame'])
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])    
    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Percentile of algorithm risk score", fontsize=25, labelpad=15)
    plt.ylabel("Comorbidity (Charlson score)", fontsize=25, labelpad=15)
    axs[0,0].set_title('MIMIC-trained model', fontsize=20, pad=10)
    axs[0,1].set_title('STARR-trained model', fontsize=20, pad=10)
    
    fig.tight_layout()
    path = os.path.join('models/outputs/charlson', '2_all_charlsonplot.jpg')   
    fig.savefig(path)
    
    
    

# ---------------------------- PLOT GROUPED BY DEMOGRAPHIC VARIABLE -------------------------------   

def charlson_grouped(listfiles, dem_var, nbplots=3):
        
    fig, axs = plt.subplots(1,nbplots, figsize=(nbplots*7, 5))
    lowess = sm.nonparametric.lowess
    labels = []
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in charlson_curves]
    quantiles = np.arange(0,100,1)
    groups = listfiles[0][dem_var].unique()
    
    for idx, listfile in enumerate(listfiles):
        ctr = 0
        for group in groups:
            name = find_map_key(dem_var, group)
            if name not in groups_excluded:
                charlson = [listfile[(listfile['prediction']*100>=quantile) & (listfile[dem_var]==group)]['comorb2'].mean() for quantile in quantiles]
                axs[idx].scatter(x=quantiles, y=charlson, s=15, marker='x', c=charlson_curves[ctr], alpha=0.7)
                z = lowess(endog=charlson, exog=quantiles, frac=0.3, it=0, return_sorted=True)
                axs[idx].plot(z[:,0], z[:,1], c=charlson_curves[ctr])
                ctr += 1
                if idx == 0:
                    labels.append(labeldict[name])
        axs[idx].legend(lines, labels, loc='upper left',fontsize=12)
    
    for ax in axs:
        ax.grid(c=colors['grid'], zorder=100)
        ax.tick_params(labelsize=12)
        ax.set_axisbelow(True)
        ax.set_xlim(-3,103)
        ax.set_ylim(-0.5,13)
        plt.setp(ax.spines.values(), color=colors['frame'])
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])    
        ax.set_xlabel("Percentile of algorithm risk score", fontsize=14, labelpad=10)
        ax.set_ylabel("Comorbidity (Charlson score)", fontsize=14, labelpad=5)
        
    axs[0].set_title('MIMIC-trained model on full STARR data', fontsize=16)
    axs[1].set_title('STARR-trained model on full STARR data', fontsize=16)
    if nbplots == 3:
        axs[2].set_title('STARR-trained model on STARR test data', fontsize=16)
    
    fig.tight_layout()
    path = os.path.join('models/outputs/charlson', str(nbplots) + '_' + dem_var + '_charlsonplot.jpg')   
    fig.savefig(path)
    
    
    
    

# ---------------------------- INDIVIDUAL PLOTS ------------------------------- 

def plot_smoothed(listfile, dem_var):
        
    f, ax = plt.subplots(figsize=(7, 5))
    lowess = sm.nonparametric.lowess
    ctr = 0
    labels = []
    quantiles = np.arange(0,100,1)
    
    for group in listfile[dem_var].unique():
        name = find_map_key(dem_var, group)
        if name not in groups_excluded:
            charlson = [listfile[(listfile['prediction']*100>=quantile) & (listfile[dem_var]==group)]['comorb2'].mean() for quantile in quantiles]
            ax.scatter(x=quantiles, y=charlson, s=15, marker='x', c=charlson_curves[ctr], alpha=0.7)
            z = lowess(endog=charlson, exog=quantiles, frac=0.3, it=0, return_sorted=True)
            ax.plot(z[:,0], z[:,1], c=charlson_curves[ctr])
            ctr += 1
            labels.append(labeldict[name])
    
    ax.grid(c=colors['grid'], zorder=100)
    ax.tick_params(labelsize=12)
    ax.set_axisbelow(True)
    ax.set_xlim(-3,103)
    ax.set_ylim(-0.5,13)
    plt.setp(ax.spines.values(), color=colors['frame'])
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in charlson_curves]
    plt.legend(lines, labels, loc='upper left',fontsize=12)
    
    plt.xlabel("Percentile of algorithm risk score", fontsize=16, labelpad=10)
    plt.ylabel("Comorbidity (Charlson score)", fontsize=16, labelpad=10)