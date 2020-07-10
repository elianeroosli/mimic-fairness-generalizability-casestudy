# +-------------------------------------------------------------------------------------------------+
# | performance.py: functions to analyze and plot performance metrics                               |
# |                                                                                                 |
# | Eliane Röösli (2020)                                                                            |
# +-------------------------------------------------------------------------------------------------+

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from models.evaluation.configs import groups_excluded, labeldict, is_public_map, colors, paths, metrics, two_curves


# read the metrics file
def metrics_csv(path):
    df = pd.read_csv(path, delimiter = ',')
    df = df[df["dem_val"] != "Selfpay"]
    df = df[df["dem_val"] != "OTHER"]
    df.set_index(keys=["dem_var", "dem_val"], inplace=True)
    return df



#------------------------------------------------ STATS --------------------------------------

# collect all data from either training or validation    
def group_data(df, group='train'):
    idx = [(col.find(group) != -1) for col in df.columns]
    return df[df.columns[idx]]


# process metrics and df for plotting
def df_processing(df, var):
    df_T = df.T
    df_T.loc["nb/1000"] = df_T.loc["nb"]/1000
    df_T.loc["nb/1000", 'None'] = 0
    df_T.drop(labels="nb", axis=0, inplace=True)
    df_T_var = df_T[['None', var]]
    df_T_var.columns = df_T_var.columns.droplevel(0)
    df_T_var.rename(columns={"None": "OVERALL"}, inplace=True)
    return df_T_var

#---------------------------------------- PLOTS: basic performance metrics --------------------------------------


# plot the metrics
def plot_metrics_basic(df, var):
    cmap = plt.cm.GnBu
    cmaplist = [cmap(int(i)) for i in np.flip(np.linspace(50, cmap.N, df.shape[1]-1),axis=0)]
    cmaplist.insert(0, (.5, .5, .5, 1.0)) # force the first color entry to be grey
    
    df.plot(kind='bar', figsize=(15,8), color=cmaplist)
    plt.ylabel('Score')
    plt.xlabel('Metrics')
    plt.title('Overview of metrics per demographic group for: ' + var)
    

#---------------------------------------- PLOTS: advanced performance metrics (with conf ints) --------------------------------------
    
    



def plot_metric_grouped(output_json, variable, data='mimic'):
    fig, axs = plt.subplots(4, 1, figsize=(9,4), sharex=True, gridspec_kw={'height_ratios': [1,3,2,4], 'hspace': 0})
    if variable in metrics:
        plt.xlabel(metrics[variable]["name"], fontsize=16, labelpad=10)
    else:
        plt.xlabel(variable, fontsize=16, labelpad=10)
    axs[1].margins(y=0.4)
    axs[2].margins(y=0.6)
    axs[3].margins(y=0.3)

    for ax in axs:
        plt.setp(ax.spines.values(), color=colors['frame'])
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])
        if variable == 'AUC of ROC':
            ax.set_xlim(0.45, 1)
            ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        if variable == 'AUC of PRC':
            ax.set_xlim(-0.01, 0.92)
            ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        if variable == 'Accuracy':
            ax.set_xlim(0.73, 0.97)
            ax.set_xticks([0.75, 0.8, 0.85, 0.9, 0.95])
        ax.label_outer()
        ax.grid(c=colors['grid'], axis='x')
        ax.tick_params(labelsize=12)
        ax.set_axisbelow(True)
    
    for idx, key in enumerate(output_json.keys()):
        # ignore n_iters
        if key == 'n_iters' or key == 'binaryIns':
            continue
        
        # special treatment for 'overall' category
        df = pd.DataFrame(output_json[key])
        if key == 'Overall':
            med = df[variable].loc['median']
            std = df[variable].loc['std']
            axs[0].hlines(y=labeldict[key], xmin=df[variable].loc['2.5% percentile'], 
                          xmax=df[variable].loc['97.5% percentile'], linewidth=1, color='grey')
            axs[0].hlines(y=labeldict[key], xmin=med-std, xmax=med+std, linewidth=3)
            axs[0].scatter(y=labeldict[key], x=df[variable].loc['median'], color="black")
            continue
        
        # rest of demographic groups
        for group in sorted(df.columns, reverse=True):
            # subgroups to ignore in analysis
            if group in groups_excluded:
                continue
            med = df[group].loc[variable]['median']
            std = df[group].loc[variable]['std']
            axs[idx-2].hlines(y=labeldict[group], xmin=df[group].loc[variable]['2.5% percentile'], 
                              xmax=df[group].loc[variable]['97.5% percentile'], linewidth=1, color='grey')
            axs[idx-2].hlines(y=labeldict[group], xmin=med-std, xmax=med+std, linewidth=3)
            axs[idx-2].scatter(y=labeldict[group], x=df[group].loc[variable]['median'], color="black")
    
    fig.tight_layout()
    path = os.path.join(paths['output'], data, paths['cv'], variable + "_confval_plot.jpg")   
    fig.savefig(path)
        
    
def plot_metrics_overall(output_json, data='mimic'):
    fig, axs = plt.subplots(4, 1, figsize=(8,3), sharex=True, gridspec_kw={'height_ratios': [2,1,2,2], 'hspace': 0})
    plt.xlabel("Score", fontsize=16, labelpad=10)
    plt.setp(axs, xticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axs[0].margins(y=0.5)
    axs[1].margins(y=2.5)
    axs[2].margins(y=0.5)
    axs[3].margins(y=0.5)
    
    for ax in axs:
        plt.setp(ax.spines.values(), color=colors['frame'])
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])
        ax.set_xlim(0.08, 1.02)
        ax.label_outer()
        ax.grid(c=colors['grid'], axis='x')
        ax.tick_params(labelsize=12)
        ax.set_axisbelow(True)
    
    df = pd.DataFrame(output_json['Overall'])
    for m in metrics.keys():
        med = df[m].loc['median']
        std = df[m].loc['std']
        axs[metrics[m]['ax']].hlines(y=metrics[m]['name'], xmin=df[m].loc['2.5% percentile'], 
                      xmax=df[m].loc['97.5% percentile'], linewidth=1, color='grey')
        axs[metrics[m]['ax']].hlines(y=metrics[m]['name'], xmin=med-std, xmax=med+std, linewidth=3)
        axs[metrics[m]['ax']].scatter(y=metrics[m]['name'], x=df[m].loc['median'], s=10, marker='o', color="black")
        
    fig.tight_layout()
    path = os.path.join(paths['output'], data, paths['cv'], "overallperformance_confval_plot.jpg")   
    fig.savefig(path)

    

#--------------------------- plot curves ----------------------------------------    


def find_csv(dir_path, variable, curve_type):
    csv_list = os.listdir(dir_path)
    variable_list = []
    for l in csv_list:
        if (l.find(variable) != -1) and (l.find(curve_type) != -1):
            if l.find("Selfpay") == -1:
                variable_list.append(l)
    return variable_list


def plot_prc(dir_path, variable, baseline):
    variable_list = find_csv(dir_path, variable, "prc")
            
    f, ax = plt.subplots(figsize=(8, 5))
    plt.setp(ax.spines.values(), color=colors['frame'])
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])
    ax.set_axisbelow(True)
    for l in variable_list:
        path_curve = os.path.join(dir_path, l)
        df = pd.read_csv(path_curve)
        label = l.split("_")[1]
        ax.plot(df["recalls"], df["precisions"], label=label, color=two_curves[0], clip_on=False)

    ax.axhline(baseline, color=two_curves[1], linestyle='--')
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.tick_params(labelsize=12)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    if variable != 'overall':
        plt.legend()
        
    if variable == 'overall':
        f.savefig('plots/prc')
    
    
def plot_roc(dir_path, variable):
    variable_list = find_csv(dir_path, variable, "roc")
            
    f, ax = plt.subplots(figsize=(8, 5))
    plt.setp(ax.spines.values(), color=colors['frame'])
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])
    ax.set_axisbelow(True)
    for l in variable_list:
        path_curve = os.path.join(dir_path, l)
        df = pd.read_csv(path_curve)
        label = l.split("_")[1]
        ax.plot(df["fpr"], df["tpr"], label=label, color=two_curves[0], clip_on=False)

    ax.plot([0,1], [0,1], color=two_curves[1], linestyle='--')
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.tick_params(labelsize=12)
    plt.xlabel("False positive rate", fontsize=16)
    plt.ylabel("True positive rate", fontsize=16)

    
    if variable != 'overall':
        plt.legend()
        
    if variable == 'overall':
        f.savefig('plots/roc')
    

def plot_roc_prc(dir_path, variable):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
        
    # ROC
    roc_list = find_csv(dir_path, variable, "roc")
    for l in roc_list:
        path_curve = os.path.join(dir_path, l)
        df = pd.read_csv(path_curve)
        label = l.split("_")[1]
        ax1.plot(df["fpr"], df["tpr"], label=label)

    ax1.set_title("RO-CURVE")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.legend()
    
    # PRC
    prc_list = find_csv(dir_path, variable, "prc")
    for l in prc_list:
        path_curve = os.path.join(dir_path, l)
        df = pd.read_csv(path_curve)
        label = l.split("_")[1]
        ax2.plot(df["recalls"], df["precisions"], label=label)

    ax2.set_title("PR-CURVE")
    ax2.set_xlabel("Recalls")
    ax2.set_ylabel("Precisions")
    fig.tight_layout(pad=3.0)

    
def plot_cutoff(path, data='mimic'):
    df = pd.read_csv(path, engine='python')
    
    f, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(df["thresholds"], df["recalls"], color=colors["val"], clip_on=False)
    ax1.set_ylabel('Recall', color=colors["val"], fontsize=16, labelpad=10)
    ax1.set_xlabel('Threshold', fontsize=16, labelpad=10)
    ax1.tick_params(labelsize=12)
    ax1.set_xlim(-0.005, 1.005)
    ax1.set_ylim(-0.005, 1.005)
    
    ax2 = ax1.twinx()
    ax2.plot(df["thresholds"], df["precisions"], color=colors["train"], clip_on=False)
    ax2.set_ylabel('Precision', color=colors["train"], fontsize=16, labelpad=10)
    ax2.set_xlim(-0.005, 1.005)
    ax2.set_ylim(-0.005, 1.005)

    hor = np.max([min(x, y) for (x, y) in zip(df['precisions'], df['recalls'])])
    ver = df['thresholds'][np.argmax([min(x, y) for (x, y) in zip(df['precisions'], df['recalls'])])]
    ax1.axhline(hor, linestyle='--', color=two_curves[1])
    ax1.axvline(ver, linestyle='--', color=two_curves[1])

    plt.setp(ax1.spines.values(), color=colors['frame'])
    plt.setp(ax2.spines.values(), color=colors['frame'])
    plt.setp([ax1.get_xticklines(), ax1.get_yticklines()], color=colors['frame'])
    plt.setp([ax2.get_xticklines(), ax2.get_yticklines()], color=colors['frame'])
    f.tight_layout()
    plt.show()
    f.savefig(os.path.join(paths['output'], data, 'plots/pr_cutoff_plot.jpg'))
