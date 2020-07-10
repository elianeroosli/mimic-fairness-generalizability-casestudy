# +-------------------------------------------------------------------------------------------------+
# | learningcurves.py: functions to analyze and plot learning curves                                |
# |                                                                                                 |
# | Eliane Röösli (2020)                                                                            |
# +-------------------------------------------------------------------------------------------------+

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from models.evaluation.configs import colors, paths

# read on epoch file
def epoch_csv(path):
    return pd.read_csv(path, delimiter = ';', index_col='epoch')

#------------------------------------------------- STATS ------------------------------------

def best_epoch(df):
    return df['val_auroc'].values.argmax()+1

# stats
def best_auroc(df):
    return np.round(df['val_auroc'][best_epoch(df)-1],6)


#--------------------------------------------------- PLOTTING ----------------------------------------------


# single plot: learning curve (auroc over epochs) 
def plot_learningcurves(df, data='mimic'):
    print('best epoch:', best_epoch(df))
    print('best auc-roc score:', best_auroc(df))
    
    epochs = range(1,len(df['train_auroc'])+1)
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    plt.plot(epochs, df['train_auroc'], c=colors['train'], label='Training')
    plt.plot(epochs, df['val_auroc'], c=colors['val'], label='Validation')
    plt.scatter(x=best_epoch(df), y=best_auroc(df), c=colors['point'])
    plt.xlabel('Trained epochs', fontsize=16)
    plt.ylabel('AUROC', fontsize=16)
    ax.tick_params(labelsize=12)
    plt.grid(c=colors['grid'])
    plt.setp(ax.spines.values(), color=colors['frame'])
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])
    plt.legend(fontsize=14)
    fig.tight_layout()
    
    fig.savefig(os.path.join(paths['output'], data, 'plots/learningcurve.jpg'))


def plot_learningcurves_2by2(dfs, df_base, data='mimic'):
    fig, axs = plt.subplots(2, 2, figsize=(10,8), gridspec_kw={'wspace': 0.0})
    if data == 'mimic':
        plt.setp(axs, ylim=(0.79,0.93), xticks=[1,25,50,75,100])
    if data == 'starr':
        plt.setp(axs, ylim=(0.73,0.97), xticks=[1,25,50,75,100], yticks=[0.75, 0.8, 0.85, 0.9, 0.95])
    epochs = range(1,len(dfs[1]['train_auroc'])+1)
    
    # PLOT: upper-left
    axs[0, 0].plot(epochs, df_base['train_auroc'], c=colors['train_base'])
    axs[0, 0].plot(epochs, df_base['val_auroc'], c=colors['val_base'])
    axs[0, 0].plot(epochs, dfs[0]['train_auroc'], c=colors['train'])
    axs[0, 0].plot(epochs, dfs[0]['val_auroc'], c=colors['val'])
    axs[0, 0].scatter(x=best_epoch(dfs[0]), y=best_auroc(dfs[0]), c=colors['point'], zorder=10)
    axs[0, 0].set_title('(a) Full demographic data', fontsize=18, pad=10)
    
    # PLOT: upper-right
    axs[0, 1].plot(epochs, df_base['train_auroc'], c=colors['train_base'])
    axs[0, 1].plot(epochs, df_base['val_auroc'], c=colors['val_base'])
    axs[0, 1].plot(epochs, dfs[1]['train_auroc'], c=colors['train'])
    axs[0, 1].plot(epochs, dfs[1]['val_auroc'], c=colors['val'])
    axs[0, 1].scatter(x=best_epoch(dfs[1]), y=best_auroc(dfs[1]), c=colors['point'], zorder=10)
    axs[0, 1].set_title('(b) Gender data only', fontsize=18, pad=10)
    
    # PLOT: lower-left
    axs[1, 0].plot(epochs, df_base['train_auroc'], c=colors['train_base'])
    axs[1, 0].plot(epochs, df_base['val_auroc'], c=colors['val_base'])
    axs[1, 0].plot(epochs, dfs[2]['train_auroc'], c=colors['train'])
    axs[1, 0].plot(epochs, dfs[2]['val_auroc'], c=colors['val'])
    axs[1, 0].scatter(x=best_epoch(dfs[2]), y=best_auroc(dfs[2]), c=colors['point'], zorder=10)
    axs[1, 0].set_title('(c) Insurance data only', fontsize=18, pad=10)
    
    # PLOT: lower-right
    axs[1, 1].plot(epochs, df_base['train_auroc'], c=colors['train_base'])
    axs[1, 1].plot(epochs, df_base['val_auroc'], c=colors['val_base'])
    axs[1, 1].plot(epochs, dfs[3]['train_auroc'], c=colors['train'])
    axs[1, 1].plot(epochs, dfs[3]['val_auroc'], c=colors['val'])
    axs[1, 1].scatter(x=best_epoch(dfs[3]), y=best_auroc(dfs[3]), c=colors['point'], zorder=10)
    axs[1, 1].set_title('(d) Ethnicity data only', fontsize=18, pad=10)
    
    # add outer subplot for labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Epochs", fontsize=25, labelpad=10)
    plt.ylabel("AUROC", fontsize=25, labelpad=20)

    # hide x labels and tick labels for top plots and y ticks for right plots
    for ax in axs.flat:
        ax.label_outer()
        ax.set_axisbelow(True)
        ax.grid(c=colors['grid'])
        plt.setp(ax.spines.values(), color=colors['frame'])
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=colors['frame'])
    axs[0,1].tick_params(left=False)
    axs[1,1].tick_params(left=False)
    axs[0,0].tick_params(bottom=False)
    axs[0,1].tick_params(bottom=False)
    
    fig.savefig(os.path.join(paths['output'], data, 'plots/comparison_plot.jpg'))
    
