import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from mimic3models.evaluation.configs import colors

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
def plot_learningcurves(df):
    print('best epoch:', best_epoch(df))
    print('best auc-roc score:', best_auroc(df))
    font = {'size'   : 16}
    matplotlib.rc('font', **font)
    
    epochs = range(1,len(df['train_auroc'])+1)
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    plt.plot(epochs, df['train_auroc'], c=colors['train'], label='Training')
    plt.plot(epochs, df['val_auroc'], c=colors['val'], label='Validation')
    plt.scatter(x=best_epoch(df), y=best_auroc(df), c=colors['point'])
    plt.xlabel('Trained Epochs')
    plt.ylabel('AUC-ROC')
    plt.grid(c=colors['grid'])
    plt.legend()
    fig.tight_layout()
    
    fig.savefig('mimic3models/in_hospital_mortality/plots/learningcurve.jpg')


def plot_learningcurves_2by2(dfs, df_base):
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    plt.setp(axs, ylim=(0.79,0.93), xticks=[1,25,50,75,100])
    axs[0, 0].grid(c=colors['grid'])
    axs[1, 0].grid(c=colors['grid'])
    axs[0, 1].grid(c=colors['grid'])
    axs[1, 1].grid(c=colors['grid'])
    epochs = range(1,len(dfs[0]['train_auroc'])+1)
    
    # PLOT: upper-right
    axs[0, 0].plot(epochs, df_base['train_auroc'], c=colors['train_base'])
    axs[0, 0].plot(epochs, df_base['val_auroc'], c=colors['val_base'])
    axs[0, 0].plot(epochs, dfs[0]['train_auroc'], c=colors['train'])
    axs[0, 0].plot(epochs, dfs[0]['val_auroc'], c=colors['val'])
    axs[0, 0].scatter(x=best_epoch(dfs[0]), y=best_auroc(dfs[0]), c=colors['point'])
    axs[0, 0].set_title('(a) Full demographic data', fontsize=18)
    
    # PLOT: upper-left
    axs[0, 1].plot(epochs, df_base['train_auroc'], c=colors['train_base'])
    axs[0, 1].plot(epochs, df_base['val_auroc'], c=colors['val_base'])
    axs[0, 1].plot(epochs, dfs[1]['train_auroc'], c=colors['train'])
    axs[0, 1].plot(epochs, dfs[1]['val_auroc'], c=colors['val'])
    axs[0, 1].scatter(x=best_epoch(dfs[1]), y=best_auroc(dfs[1]), c=colors['point'])
    axs[0, 1].set_title('(b) Gender data only', fontsize=18)
    
    # PLOT: lower-right
    axs[1, 0].plot(epochs, df_base['train_auroc'], c=colors['train_base'])
    axs[1, 0].plot(epochs, df_base['val_auroc'], c=colors['val_base'])
    axs[1, 0].plot(epochs, dfs[2]['train_auroc'], c=colors['train'])
    axs[1, 0].plot(epochs, dfs[2]['val_auroc'], c=colors['val'])
    axs[1, 0].scatter(x=best_epoch(dfs[2]), y=best_auroc(dfs[2]), c=colors['point'])
    axs[1, 0].set_title('(c) Insurance data only', fontsize=18)
    
    # PLOT: lower-left
    axs[1, 1].plot(epochs, df_base['train_auroc'], c=colors['train_base'])
    axs[1, 1].plot(epochs, df_base['val_auroc'], c=colors['val_base'])
    axs[1, 1].plot(epochs, dfs[3]['train_auroc'], c=colors['train'])
    axs[1, 1].plot(epochs, dfs[3]['val_auroc'], c=colors['val'])
    axs[1, 1].scatter(x=best_epoch(dfs[3]), y=best_auroc(dfs[3]), c=colors['point'])
    axs[1, 1].set_title('(d) Ethnicity data only', fontsize=18)
    
    # add outer subplot for labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Epochs", fontsize=25, labelpad=10)
    plt.ylabel("AUC-ROC", fontsize=25, labelpad=20)

    # hide x labels and tick labels for top plots and y ticks for right plots
    for ax in axs.flat:
        ax.label_outer()
    
    fig.savefig('mimic3models/in_hospital_mortality/plots/comparison_plot.jpg')
    
