# +-------------------------------------------------------------------------------------------------+
# | calibration.py: functions to analyze and plot calibration                                       |
# |                                                                                                 |
# | Eliane Röösli (2020)                                                                            |
# +-------------------------------------------------------------------------------------------------+

import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats as ss
from scipy.stats import chi2

from benchmarks.common_tools import find_map_key
from models.evaluation.configs import groups_excluded, labeldict, is_public_map, colors, paths


# load results csv file
def results_csv(path):
    df = pd.read_csv(path, delimiter = ',')
    df.set_index("stay", inplace=True)
    df.sort_values(by=["prediction"], axis=0, inplace=True)
    return df

#--------------------------------------------- STATS -----------------------------------------

def hl_test(data, g, verbose=True):
    '''
    Hosmer-Lemeshow test to judge the goodness of fit for binary data

    Input: 
    - data: predictions and true outcomes 
    - g: number of bins/groups
    
    Output: 
    - hltest: value of test statistic
    - pval: p-value associated with hltest
    '''
    
    data_st = data.sort_values('prediction')
    data_st['dcl'] = pd.qcut(data_st['prediction'], g)
    
    # ys: number of expected positive cases, yn: expected negative cases, yt: total number of cases
    ys = data_st['y_true'].groupby(data_st.dcl).sum()
    yt = data_st['y_true'].groupby(data_st.dcl).count()
    yn = yt - ys
    
    # yps: number of predicted positive cases, ypn: predicted negative cases
    yps = data_st['prediction'].groupby(data_st.dcl).apply(lambda x: x.mean()*x.shape[0]) 
    ypn = data_st['prediction'].groupby(data_st.dcl).apply(lambda x: (1-x.mean())*x.shape[0])
    
    # test statistic
    hltest = ( ((ys - yps)**2 / yps) + ((yn - ypn)**2 / ypn) ).sum()
    df = g-2
    pval = 1 - chi2.cdf(hltest, df)
    
    if verbose:
        print('HL-chi2({}): {}, p-value: {}'.format(df, hltest, pval))
    
    return hltest, pval



# BINNING: group samples into risk groups based on the prediction and compute mean predicted and observed mortality risk per group
def risk_groups(df, g, option='quantiles'):
    df.reset_index(drop=True, inplace=True)
    
    # three different grouping options: quantiles, percentiles or exponential quantiles
    if option == 'percentiles':
        percentage_size = 100/g
        df['risk_group'] = np.ceil(df["prediction"]*100/percentage_size).astype(int)
    elif option == 'exp_quantiles':
        df['risk_group'] = pd.qcut(df['prediction'], np.linspace(0.0, 1.0, num=g+1)**0.20, labels=False)
    else:
        group_size = df.shape[0]/g
        df['risk_group'] = np.ceil((df.index+1)/group_size).astype(int)
    
    # group by risk category and calculate group stats
    df_grouped = df.groupby("risk_group")[["prediction", "y_true"]].mean()
    df_grouped['n_obs'] = df.groupby("risk_group")['y_true'].count()
    df_grouped['counts'] = df.groupby("risk_group")['y_true'].sum()
    return df_grouped.reset_index(drop=True)


# analyse calibration in demographic groups by doing the risk calibration analysis for each group individually
def calibration_dem(df, var, g=10):
    if var == 'binaryIns':
        is_public = [is_public_map[str(x)] for x in df["Insurance"]]
        df["binaryIns"] = is_public
        df = df[df["binaryIns"] != 2]
    values = df[var].unique()
    dfs = []
    labels = []
    for val in values:
        val_def = df[df[var] == val]
        dfs.append(risk_groups(val_def, g))
        labels.append(find_map_key(var, int(val)))
    return dfs, labels


#--------------------------------------------- PLOTS: basic calibration plots -----------------------------------------

# scatter plot
def plot_calibration(dfs, labels):
    f, ax = plt.subplots(figsize=(8, 8))
    x_max, y_max = 0
    for i, df in enumerate(dfs):
        ax.scatter(x=df["prediction"], y=df["y_true"], s=70, label=labels[i])
        x_max = np.maximum(df["prediction"].max(), x_max)
        y_max = np.maximum(df["y_true"].max(), y_max)
        
    ax.set(xlim=(dfs[0]["prediction"].min()-0.05, x_max+0.05), ylim=(dfs[0]["y_true"].min()-0.05, y_max+0.05))
    diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    
    plt.legend()
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.show()
    

# connected scatter plot
def connected_plot_calibration(dfs, labels):
    f, ax = plt.subplots(figsize=(8, 8))
    x_max = 0
    y_max = 0
    for i, df in enumerate(dfs):
        ax.plot(df["prediction"], df["y_true"], marker='o', markersize=2, label=labels[i])
        x_max = np.maximum(df["prediction"].max(), x_max)
        y_max = np.maximum(df["y_true"].max(), y_max)
    
    ax.set(xlim=(dfs[0]["prediction"].min()-0.05, x_max+0.05), ylim=(dfs[0]["y_true"].min()-0.05, y_max+0.05))
    diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    
    plt.legend()
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.show()

  
    
def side_by_side_calibration(dfs, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    x_max = 0
    y_max = 0
    for i, df in enumerate(dfs):
        ax1.scatter(x=df["prediction"], y=df["y_true"], s=70, label=labels[i])
        ax2.plot(df["prediction"], df["y_true"], marker='o', markersize=4, ls=':', lw=2, label=labels[i])
        x_max = np.maximum(df["prediction"].max(), x_max)
        y_max = np.maximum(df["y_true"].max(), y_max)

    ax1.set(xlim=(dfs[0]["prediction"].min()-0.05, x_max+0.05), ylim=(dfs[0]["y_true"].min()-0.05, y_max+0.05))
    ax2.set(xlim=(dfs[0]["prediction"].min()-0.05, x_max+0.05), ylim=(dfs[0]["y_true"].min()-0.05, y_max+0.05))
    diag_line, = ax1.plot(ax1.get_xlim(), ax1.get_ylim(), ls="--", c=".7")
    diag_line, = ax2.plot(ax2.get_xlim(), ax2.get_ylim(), ls="--", c=".7")
    
    ax1.legend()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Predicted probability", fontsize=20, labelpad=10)
    plt.ylabel("Observed frequency", fontsize=20, labelpad=20)
    fig.tight_layout(pad=3.0)
    

#--------------------------------------------- PLOTS: calibration-in-the-large -----------------------------------------


def plot_calibration_mean(output_json, data='mimic'):
    fig, axs = plt.subplots(4, 1, figsize=(9,5), sharex=True, gridspec_kw={'height_ratios': [1,3,2,4], 'hspace': 0})
    plt.xlabel("Average risk score", fontsize=16, labelpad=10)
    axs[1].margins(y=0.10)
    axs[2].margins(y=0.15)
    axs[3].margins(y=0.05)
    
    variables = ['Average observed risk', 'Average predicted risk']
    for idx, key in enumerate(output_json.keys()):
        if key == 'n_iters' or key == 'binaryIns':
            continue
        
        df = pd.DataFrame(output_json[key])
        if key == 'Overall':
            group = key
            # invisible padding for plot
            axs[idx-1].scatter(y=labeldict[group]+str(0.1),x=0.1, s=0)
            axs[idx-1].scatter(y=labeldict[group]+str(0.2),x=0.1, s=0)
            for v, variable in enumerate(variables):
                med = df[variable].loc['median']
                std = df[variable].loc['std']
                axs[0].hlines(y=labeldict[group]+str(v), xmin=df[variable].loc['2.5% percentile'], 
                              xmax=df[variable].loc['97.5% percentile'], linewidth=1, color='grey')
                axs[0].hlines(y=labeldict[group]+str(v), xmin=med-std, xmax=med+std, linewidth=3, color=colors['cint'][v])
                axs[0].scatter(y=labeldict[group]+str(v), x=med, color=colors['cint'][v], label=variable)
                # invisible padding for plot
                if v==0:
                    axs[idx-1].scatter(y=labeldict[group],x=0.1, s=0)
                if v==1:
                    axs[idx-1].scatter(y=labeldict[group]+str(v+0.1),x=0.1, s=0)
                    axs[idx-1].scatter(y=labeldict[group]+str(v+0.2),x=0.1, s=0)
            continue
        
        for group in sorted(df.columns, reverse=True):
            if group in groups_excluded:
                continue
            # invisible padding for plot
            axs[idx-2].scatter(y=labeldict[group]+str(0.1),x=0.1, s=0)
            axs[idx-2].scatter(y=labeldict[group]+str(0.2),x=0.1, s=0)
            for v, variable in enumerate(variables):
                med = df[group].loc[variable]['median']
                std = df[group].loc[variable]['std']
                axs[idx-2].hlines(y=labeldict[group]+str(v), xmin=df[group].loc[variable]['2.5% percentile'], 
                                  xmax=df[group].loc[variable]['97.5% percentile'], linewidth=1, color='grey')
                axs[idx-2].hlines(y=labeldict[group]+str(v), xmin=med-std, xmax=med+std, linewidth=3, color=colors['cint'][v])
                axs[idx-2].scatter(y=labeldict[group]+str(v), x=med, color=colors['cint'][v])
                # invisible padding for plot
                if v==0:
                    axs[idx-2].scatter(y=labeldict[group],x=0.1, s=0)
                if v==1:
                    axs[idx-2].scatter(y=labeldict[group]+str(v+0.1),x=0.1, s=0)
                    axs[idx-2].scatter(y=labeldict[group]+str(v+0.2),x=0.1, s=0)
    
    # fit the axis
    for ax in axs:
        plt.setp(ax.spines.values(), color=colors['frame'])
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])
        ax.label_outer()
        ax.grid(c=colors['grid'], axis='x')
        ax.tick_params(labelsize=12)
        ax.set_xlim(0.01, 0.24)
        ax.set_xticks([0.05, 0.1, 0.15, 0.2])
        y_ticks = [tick for tick in ax.get_yticks() if tick in [3,10,17,24,31]]
        ax.set_yticks(y_ticks)
        ax.set_axisbelow(True)
    
    # plot legend (reverse order to match plot)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper right', prop={'size': 12})
    
    fig.tight_layout()
    path = os.path.join(paths['output'], data, paths['cb'], "confval_calibration_plot.jpg")   
    fig.savefig(path)
    
    
def plot_calibration_mean_strat(output_json, data='mimic'):
    fig, axs = plt.subplots(4, 1, figsize=(9,4.5), sharex=True, gridspec_kw={'height_ratios': [1,3,2,4], 'hspace': 0})
    plt.xlabel("Average risk score", fontsize=16, labelpad=10)
    axs[1].margins(y=0.4)
    axs[2].margins(y=0.6)
    axs[3].margins(y=0.3)

    variables = ['Average observed risk', 'Average predicted risk']
    for idx, key in enumerate(output_json.keys()):
        if key == 'n_iters' or key == 'binaryIns':
            continue
        
        df = pd.DataFrame(output_json[key])
        if key == 'Overall':
            group = key
            for v, variable in enumerate(variables):
                med = df[variable].loc['median']
                std = df[variable].loc['std']
                axs[0].scatter(y=labeldict[group], x=med, color=colors['cint'][v], label=variable)
                if variable == 'Average predicted risk':
                    axs[0].hlines(y=labeldict[group], xmin=df[variable].loc['2.5% percentile'], 
                                  xmax=df[variable].loc['97.5% percentile'], linewidth=1, color='grey')
                    axs[0].hlines(y=labeldict[group], xmin=med-std, xmax=med+std, linewidth=3, color=colors['cint'][v])
            continue
        
        for group in sorted(df.columns, reverse=True):
            if group in groups_excluded:
                continue
            for v, variable in enumerate(variables):
                med = df[group].loc[variable]['median']
                std = df[group].loc[variable]['std']
                axs[idx-2].scatter(y=labeldict[group], x=med, color=colors['cint'][v])
                if variable == 'Average predicted risk':
                    axs[idx-2].hlines(y=labeldict[group], xmin=df[group].loc[variable]['2.5% percentile'], 
                                      xmax=df[group].loc[variable]['97.5% percentile'], linewidth=1, color='grey')
                    axs[idx-2].hlines(y=labeldict[group], xmin=med-std, xmax=med+std, linewidth=3, color=colors['cint'][v])

    # fit the axis
    for ax in axs:
        plt.setp(ax.spines.values(), color=colors['frame'])
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])
        ax.label_outer()
        ax.grid(c=colors['grid'], axis='x')
        ax.tick_params(labelsize=12)
        #y_ticks = [tick for tick in ax.get_yticks() if tick in [3,10,17,24,31]]
        #ax.set_yticks(y_ticks)
        ax.set_axisbelow(True)
    
    # plot legend (reverse order to match plot)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper right', prop={'size': 12})
    
    fig.tight_layout()
    path = os.path.join(paths['output'], data, paths['cb'], "confval_calibration_plot_strat.jpg")   
    fig.savefig(path)
    
    
    
def plot_calibration_diffs(output_json, data='mimic'):
    fig, axs = plt.subplots(4, 1, figsize=(9,4), sharex=True, gridspec_kw={'height_ratios': [1,3,2,4], 'hspace': 0})
    plt.xlabel("Predicted minus observed average risk", fontsize=16, labelpad=10)
    axs[1].margins(y=0.4)
    axs[2].margins(y=0.6)
    axs[3].margins(y=0.3)

    variable = 'Difference in average predicted and observed risk'
    for idx, key in enumerate(output_json.keys()):
        if key == 'n_iters' or key == 'binaryIns':
            continue
        
        df = pd.DataFrame(output_json[key])
        if key == 'Overall':
            group = key
            med = df[variable].loc['median']
            std = df[variable].loc['std']
            axs[0].hlines(y=labeldict[group], xmin=df[variable].loc['2.5% percentile'], 
                          xmax=df[variable].loc['97.5% percentile'], linewidth=1, color='grey')
            axs[0].hlines(y=labeldict[group], xmin=med-std, xmax=med+std, linewidth=3)
            axs[0].scatter(y=labeldict[group], x=med, color='black')
            continue
        
        for group in sorted(df.columns, reverse=True):
            if group in groups_excluded:
                continue
            med = df[group].loc[variable]['median']
            std = df[group].loc[variable]['std']
            axs[idx-2].hlines(y=labeldict[group], xmin=df[group].loc[variable]['2.5% percentile'], 
                              xmax=df[group].loc[variable]['97.5% percentile'], linewidth=1, color='grey')
            axs[idx-2].hlines(y=labeldict[group], xmin=med-std, xmax=med+std, linewidth=3)
            axs[idx-2].scatter(y=labeldict[group], x=med, color='black')

    
    # fit the axis
    for ax in axs:
        plt.setp(ax.spines.values(), color=colors['frame'])
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])
        ax.label_outer()
        ax.grid(c=colors['grid'], axis='x')
        ax.tick_params(labelsize=12)
        ax.set_xlim(-0.12, 0.12)
        ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
        ax.axvline(x=0, color='darkred', linestyle='--', zorder=10)
        ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(paths['output'], data, paths['cb'], "confval_calibration_diffs_plot.jpg")   
    fig.savefig(path)
    
    

    
#--------------------------------------------- PLOTS: advanced calibration and frequency plots -----------------------------------------
    
def calibration_overall(df, name, g=10, option='exp_quantiles', data='mimic', title=''):
    fig, axs = plt.subplots(3, 1, figsize=(6,6.5), sharex=True, gridspec_kw={'height_ratios': [6,0.8,1.2], 'hspace': 0})
    
    # PLOT 0
    axs[0].plot((0,1), (0,1), ls="--", c="#dbdbdb")
    # lowess smoothing
    lowess = sm.nonparametric.lowess
    z = lowess(endog=df['y_true'], exog=df['prediction'], frac=0.5, it=0, return_sorted=True)
    axs[0].plot(z[:,0], z[:,1], c='#34a84d', label='LOWESS')
    # build risk groups and plot with confidence intervals
    df_grouped = risk_groups(df, g, option)
    axs[0].plot(df_grouped["prediction"], df_grouped["y_true"], ls=':', lw=2, 
                marker='.', markerfacecolor='black', markeredgecolor='black', markersize=7, c='#34c0d1', label='HL groups')
    wilson_confints = ss.proportion.proportion_confint(df_grouped['counts'], df_grouped['n_obs'], alpha=0.05, method='wilson')
    for idx, ci_low in enumerate(wilson_confints[0]):
        axs[0].axvline(x=df_grouped['prediction'][idx], ymin=ci_low, ymax=wilson_confints[1][idx])
    axs[0].set_ylim(0,1)
    
    # PLOT 1
    df_event = df[df['y_true']==1]
    axs[1].hist(df_event['prediction'], range=(0,1), color='#696969', bins=200, bottom=1)
    axs[1].set_yscale("log")
    axs[1].set_ylim(top=10**2)
    axs[1].set_yticks([10])
    axs[1].set_yticklabels([10])
    
    # PLOT 2
    df_noevent = df[df['y_true']==0]
    axs[2].hist(df_noevent['prediction'], range=(0,1), color='#858585', bins=200, bottom=1)
    axs[2].invert_yaxis()
    axs[2].set_yscale("log")
    axs[2].set_ylim(bottom=6*10**2)
    axs[2].set_yticks([10, 100])
    axs[2].set_yticklabels([10,100])
    if data == 'starr_ext' and name == 'overall':
        axs[2].set_ylim(bottom=2*10**3)
        axs[2].set_yticks([10, 100, 1000])
        axs[2].set_yticklabels([10, 100, 1000])
    
    # fit the axis
    frame ='#878787'
    axs[0].legend(fontsize=14)
    plt.xlim(0,1)
    plt.xlabel("Predicted probability", fontsize=18, labelpad=10)
    axs[0].set_ylabel("Observed frequency", fontsize=16, labelpad=10)
    axs[2].set_ylabel("Counts", fontsize=16, labelpad=10)
    axs[2].yaxis.set_label_coords(-0.1,1)
    
    for ax in axs:
        plt.setp(ax.spines.values(), color=colors['frame'])
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])
        ax.grid(c=colors['grid'], axis='x')
        ax.tick_params(labelsize=12)
        ax.set_axisbelow(True)
        ax.axvline(x=0.5, color='darkred', linewidth=0.5, zorder=10)

    axs[0].grid(c=colors['grid'], axis='y')   
    axs[1].spines['bottom'].set_linewidth(0) 
    axs[2].spines['top'].set_color('#c2c2c2')
    axs[1].text(0.88, 10, "Event", fontsize=12)
    axs[2].text(0.8, 100, "Non-Event", fontsize=12)
    
    if title != '':
        fig.suptitle(title, fontsize=22)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
    
    path = os.path.join(paths['output'], data, paths['cb'], 'individual', name + "_calibration_overall.jpg")   
    fig.savefig(path)
    

    
