import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

from mimic3benchmark.preprocessing import find_map_key

#font = {'size'   : 18}

#matplotlib.rc('font', **font)



#------------------------------- analyse TRAINING AND VALIDATION EPOCHS --------------------------------------

# read on epoch file
def epoch_csv(path):
    return pd.read_csv(path, delimiter = ';', index_col='epoch')

def best_epoch(df):
    return df['val_auroc'].values.argmax()+1

def best_auroc(df):
    return np.round(df['val_auroc'][best_epoch(df)-1],6)


# plot auroc over epochs 
def plot_epochs(df):
    print('best epoch:', best_epoch(df))
    print('best auc-roc score:', best_auroc(df))
    grid ='#f0f0f0'
    font = {'size'   : 16}

    matplotlib.rc('font', **font)
    
    
    epochs = range(1,len(df['train_auroc'])+1)
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    #fig = plt.figure(figsize=[7,5])
    plt.plot(epochs, df['train_auroc'], c='darkgreen', label='Training')
    plt.plot(epochs, df['val_auroc'], c='darkred', label='Validation')
    plt.scatter(x=best_epoch(df), y=best_auroc(df), c='darkslategrey')
    plt.xlabel('Trained Epochs')
    plt.ylabel('AUC-ROC')
    plt.grid(c=grid)
    plt.legend()
    fig.tight_layout()
    
    fig.savefig('mimic3models/in_hospital_mortality/plots/learningcurve.jpg')


def plot_2by2(dfs, df_base):
    
    # colors for curves
    val = '#7d3737'
    train = '#3c6350'
    val_base = '#d4c3c3'
    train_base = '#b4d4c5'
    point = '#696969'
    grid ='#f0f0f0'
    
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    plt.setp(axs, ylim=(0.79,0.93), xticks=[1,25,50,75,100])
    axs[0, 0].grid(c=grid)
    axs[1, 0].grid(c=grid)
    axs[0, 1].grid(c=grid)
    axs[1, 1].grid(c=grid)
    
    epochs = range(1,len(dfs[0]['train_auroc'])+1)
    
    axs[0, 0].plot(epochs, df_base['train_auroc'], c=train_base)
    axs[0, 0].plot(epochs, df_base['val_auroc'], c=val_base)
    axs[0, 0].plot(epochs, dfs[0]['train_auroc'], c=train)
    axs[0, 0].plot(epochs, dfs[0]['val_auroc'], c=val)
    axs[0, 0].scatter(x=best_epoch(dfs[0]), y=best_auroc(dfs[0]), c=point)
    axs[0, 0].set_title('(a) Full demographic data', fontsize=18)
    
    axs[0, 1].plot(epochs, df_base['train_auroc'], c=train_base)
    axs[0, 1].plot(epochs, df_base['val_auroc'], c=val_base)
    axs[0, 1].plot(epochs, dfs[1]['train_auroc'], c=train)
    axs[0, 1].plot(epochs, dfs[1]['val_auroc'], c=val)
    axs[0, 1].scatter(x=best_epoch(dfs[1]), y=best_auroc(dfs[1]), c=point)
    axs[0, 1].set_title('(b) Gender data only', fontsize=18)
    
    axs[1, 0].plot(epochs, df_base['train_auroc'], c=train_base)
    axs[1, 0].plot(epochs, df_base['val_auroc'], c=val_base)
    axs[1, 0].plot(epochs, dfs[2]['train_auroc'], c=train)
    axs[1, 0].plot(epochs, dfs[2]['val_auroc'], c=val)
    axs[1, 0].scatter(x=best_epoch(dfs[2]), y=best_auroc(dfs[2]), c=point)
    axs[1, 0].set_title('(c) Insurance data only', fontsize=18)
    
    axs[1, 1].plot(epochs, df_base['train_auroc'], c=train_base)
    axs[1, 1].plot(epochs, df_base['val_auroc'], c=val_base)
    axs[1, 1].plot(epochs, dfs[3]['train_auroc'], c=train)
    axs[1, 1].plot(epochs, dfs[3]['val_auroc'], c=val)
    axs[1, 1].scatter(x=best_epoch(dfs[3]), y=best_auroc(dfs[3]), c=point)
    axs[1, 1].set_title('(d) Ethnicity data only', fontsize=18)
    
    # add outer subplot for labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Epochs", fontsize=25, labelpad=10)
    plt.ylabel("AUC-ROC", fontsize=25, labelpad=20)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
    fig.savefig('mimic3models/in_hospital_mortality/plots/comparison_plot.jpg')
    
    
#------------------------------- analyse CALIBRATION --------------------------------------

def hl_test(data, g, verbose=True):
    '''
    Hosmer-Lemeshow test to judge the goodness of fit for binary data

    Input: dataframe(data), integer(num of subgroups divided)
    
    Output: float
    '''
    data_st = data.sort_values('prediction')
    data_st['dcl'] = pd.qcut(data_st['prediction'], g)
    
    # ys: number of expected positive cases, yn: expected negative cases, yt: total number of cases
    ys = data_st['y_true'].groupby(data_st.dcl).sum()
    yt = data_st['y_true'].groupby(data_st.dcl).count()
    yn = yt - ys
    
    # yps: number of predicted positive cases, ypn: predicted negative cases
    yps = data_st['prediction'].groupby(data_st.dcl).apply(lambda x: x.mean()*x.shape[0]) # same could be done by summing the probabilities
    ypn = data_st['prediction'].groupby(data_st.dcl).apply(lambda x: (1-x.mean())*x.shape[0])
    
    # test statistic
    hltest = ( ((ys - yps)**2 / yps) + ((yn - ypn)**2 / ypn) ).sum()
    df = g-2
    pval = 1 - chi2.cdf(hltest, df)
    if verbose:
        print('HL-chi2({}): {}, p-value: {}'.format(df, hltest, pval))
    return g, hltest, pval


# load results csv file
def results_csv(path):
    df = pd.read_csv(path, delimiter = ',')
    df.set_index("stay", inplace=True)
    df.sort_values(by=["prediction"], axis=0, inplace=True)
    return df


# group samples into risk groups based on the prediction and compute mean predicted and observed mortality risk per group
def risk_groups(df, g):
    df.reset_index(drop=True, inplace=True)
    group_size = df.shape[0]/g
    df['risk_group'] = np.ceil((df.index+1)/group_size).astype(int)
    return df.groupby("risk_group")[["prediction", "y_true"]].mean()


# analyse calibration in demographic groups by doing the risk calibration analysis for each group individually
def calibration_dem(df, var, g=10):
    if var == 'binaryIns':
        is_public_map = {"1": 1, "2": 1, "3": 1, "4": 0, "5": 2}
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


# scatter plot
def plot_calibration(dfs, labels):
    f, ax = plt.subplots(figsize=(8, 8))
    x_max = 0
    y_max = 0
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
    
   # add outer subplot for labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Predicted probability", fontsize=20, labelpad=10)
    plt.ylabel("Observed frequency", fontsize=20, labelpad=20)
    
    fig.tight_layout(pad=3.0)
    
#------------------------------- analyse METRICS --------------------------------------



# read the metrics file
def metrics_csv(path):
    df = pd.read_csv(path, delimiter = ',')
    df = df[df["dem_val"] != "Selfpay"]
    df = df[df["dem_val"] != "OTHER"]
    df.set_index(keys=["dem_var", "dem_val"], inplace=True)
    return df


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


# plot the metrics
def plot_metrics(df, var):
    cmap = plt.cm.GnBu
    cmaplist = [cmap(int(i)) for i in np.flip(np.linspace(50, cmap.N, df.shape[1]-1),axis=0)]
    # force the first color entry to be grey
    cmaplist.insert(0, (.5, .5, .5, 1.0))
    
    df.plot(kind='bar', figsize=(15,8), color=cmaplist)
    plt.ylabel('Score')
    plt.xlabel('Metrics')
    plt.title('Overview of metrics per demographic group for: ' + var)
    

    
labeldict = {'Overall': 'Total', 'Public': 'Public Insurance', 'Private': 'Private Insurance', 'M': 'Male', 'F': 'Female', 
              'ASIAN': 'Asian', 'HISPANIC': 'Hispanic/Latino', 'BLACK': 'Black', 'WHITE': 'White'}

def plot_intervals_structured(output_json, variable):
    fig, axs = plt.subplots(4, 1, figsize=(9,4), sharex=True, gridspec_kw={'height_ratios': [1,2,2,4], 'hspace': 0})
    plt.xlabel(variable, fontsize=16, labelpad=10)
    axs[1].margins(y=0.5)
    axs[2].margins(y=0.5)
    axs[3].margins(y=0.2)
    
    grid ='#fafafa'
    frame ='#c4c3c2'
    for ax in axs:
        plt.setp(ax.spines.values(), color=frame)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=frame)
        ax.label_outer()
        ax.grid(c=grid, axis='x')
        ax.tick_params(labelsize=12)
        ax.set_axisbelow(True)
        
    for idx, key in enumerate(output_json.keys()):
        if key == 'n_iters':
            continue
        
        df = pd.DataFrame(output_json[key])
        if key == 'Overall':
            group = key
            med = df[variable].loc['median']
            std = df[variable].loc['std']
            axs[0].hlines(y=labeldict[group], xmin=df[variable].loc['2.5% percentile'], 
                          xmax=df[variable].loc['97.5% percentile'], linewidth=1, color='grey')
            axs[0].hlines(y=labeldict[group], xmin=med-std, xmax=med+std, linewidth=3)
            axs[0].scatter(y=labeldict[group], x=df[variable].loc['median'], color="black")
            continue
        
        for group in df.columns:
            if (group == "Selfpay") | (group == 'OTHER'):
                continue
            med = df[group].loc[variable]['median']
            std = df[group].loc[variable]['std']
            axs[idx-1].hlines(y=labeldict[group], xmin=df[group].loc[variable]['2.5% percentile'], 
                              xmax=df[group].loc[variable]['97.5% percentile'], linewidth=1, color='grey')
            axs[idx-1].hlines(y=labeldict[group], xmin=med-std, xmax=med+std, linewidth=3)
            axs[idx-1].scatter(y=labeldict[group], x=df[group].loc[variable]['median'], color="black")
    
    fig.tight_layout()
    path = os.path.join('mimic3models/in_hospital_mortality/predictions/confvals', variable + "_confval_plot.jpg")   
    fig.savefig(path)
    

    
def plot_calibration_intervals(output_json):
    fig, axs = plt.subplots(4, 1, figsize=(9,6), sharex=True, gridspec_kw={'height_ratios': [1,2,2,4], 'hspace': 0})
    plt.xlabel("Average risk score", fontsize=16, labelpad=10)
    axs[1].margins(y=0.15)
    axs[2].margins(y=0.15)
    axs[3].margins(y=0.05)
    
    grid ='#fafafa'
    frame ='#c4c3c2'
    intervals = ['#3c6350', '#7d3737']

    variables = ['Average observed risk', 'Average predicted risk']
    for idx, key in enumerate(output_json.keys()):
        if key == 'n_iters':
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
                axs[0].hlines(y=labeldict[group]+str(v), xmin=med-std, xmax=med+std, linewidth=3, color=intervals[v])
                axs[0].scatter(y=labeldict[group]+str(v), x=med, color=intervals[v], label=variable)
                # invisible padding for plot
                if v==0:
                    axs[idx-1].scatter(y=labeldict[group],x=0.1, s=0)
                if v==1:
                    axs[idx-1].scatter(y=labeldict[group]+str(v+0.1),x=0.1, s=0)
                    axs[idx-1].scatter(y=labeldict[group]+str(v+0.2),x=0.1, s=0)
            continue
        
        for group in df.columns:
            if (group == "Selfpay") | (group == 'OTHER'):
                continue
            # invisible padding for plot
            axs[idx-1].scatter(y=labeldict[group]+str(0.1),x=0.1, s=0)
            axs[idx-1].scatter(y=labeldict[group]+str(0.2),x=0.1, s=0)
            for v, variable in enumerate(variables):
                med = df[group].loc[variable]['median']
                std = df[group].loc[variable]['std']
                axs[idx-1].hlines(y=labeldict[group]+str(v), xmin=df[group].loc[variable]['2.5% percentile'], 
                                  xmax=df[group].loc[variable]['97.5% percentile'], linewidth=1, color='grey')
                axs[idx-1].hlines(y=labeldict[group]+str(v), xmin=med-std, xmax=med+std, linewidth=3, color=intervals[v])
                axs[idx-1].scatter(y=labeldict[group]+str(v), x=med, color=intervals[v])
                # invisible padding for plot
                if v==0:
                    axs[idx-1].scatter(y=labeldict[group],x=0.1, s=0)
                if v==1:
                    axs[idx-1].scatter(y=labeldict[group]+str(v+0.1),x=0.1, s=0)
                    axs[idx-1].scatter(y=labeldict[group]+str(v+0.2),x=0.1, s=0)
    
    # fit the axis
    for ax in axs:
        plt.setp(ax.spines.values(), color=frame)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=frame)
        ax.label_outer()
        ax.grid(c=grid, axis='x')
        ax.tick_params(labelsize=12)
        y_ticks = [tick for tick in ax.get_yticks() if tick in [3,10,17,24,31]]
        ax.set_yticks(y_ticks)
        ax.set_axisbelow(True)
    
    # plot legend (reverse order to match plot)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper right', prop={'size': 12})
    
    fig.tight_layout()
    path = os.path.join('mimic3models/in_hospital_mortality/predictions/confvals', "confval_calibration_plot.jpg")   
    fig.savefig(path)
    

    
    
def plot_calibration_diffs(output_json):
    fig, axs = plt.subplots(4, 1, figsize=(9,5), sharex=True, gridspec_kw={'height_ratios': [1,2,2,4], 'hspace': 0})
    plt.xlabel("Predicted minus observed average risk", fontsize=16, labelpad=10)
    axs[1].margins(y=0.5)
    axs[2].margins(y=0.5)
    axs[3].margins(y=0.2)
    grid ='#fafafa'
    frame ='#c4c3c2'

    variable = 'Difference in average predicted and observed risk'
    for idx, key in enumerate(output_json.keys()):
        if key == 'n_iters':
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
        
        for group in df.columns:
            if (group == "Selfpay") | (group == 'OTHER'):
                continue
            med = df[group].loc[variable]['median']
            std = df[group].loc[variable]['std']
            axs[idx-1].hlines(y=labeldict[group], xmin=df[group].loc[variable]['2.5% percentile'], 
                              xmax=df[group].loc[variable]['97.5% percentile'], linewidth=1, color='grey')
            axs[idx-1].hlines(y=labeldict[group], xmin=med-std, xmax=med+std, linewidth=3)
            axs[idx-1].scatter(y=labeldict[group], x=med, color='black')

    
    # fit the axis
    for ax in axs:
        plt.setp(ax.spines.values(), color=frame)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=frame)
        ax.label_outer()
        ax.grid(c=grid, axis='x')
        ax.tick_params(labelsize=12)
        ax.axvline(x=0, color='darkred', linestyle='--', zorder=10)
        ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join('mimic3models/in_hospital_mortality/predictions/confvals', "confval_calibration_diffs_plot.jpg")   
    fig.savefig(path)


map_ = {'AUC of ROC':0,
           'AUC of PRC':0,
           'Accuracy':1, 
           'Precision0':2,
           'Precision1':2, 
           'Recall0':3, 
           'Recall1':3}

label_metrics = {'AUC of ROC': 'AUC of ROC',
           'AUC of PRC': 'AUC of PRC',
           'Accuracy': 'Accuracy', 
           'Precision0': 'Precision Non-Event',
           'Precision1': 'Precision Event', 
           'Recall0': 'Recall Non-Event', 
           'Recall1': 'Recall Event'}
    
def plot_performance_overall(output_json):
    fig, axs = plt.subplots(4, 1, figsize=(8,3), sharex=True, gridspec_kw={'height_ratios': [2,1,2,2], 'hspace': 0})
    plt.xlabel("Score", fontsize=16, labelpad=10)
    axs[0].margins(y=0.5)
    axs[1].margins(y=2.5)
    axs[2].margins(y=0.5)
    axs[3].margins(y=0.5)
    
    grid ='#fafafa'
    frame ='#c4c3c2'
    for ax in axs:
        plt.setp(ax.spines.values(), color=frame)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=frame)
        ax.label_outer()
        ax.grid(c=grid, axis='x')
        ax.tick_params(labelsize=12)
        ax.set_axisbelow(True)
    
    metrics = ['AUC of ROC',
           'AUC of PRC',
           'Accuracy', 
           'Precision0',
           'Precision1', 
           'Recall0', 
           'Recall1']
    
    df = pd.DataFrame(output_json['Overall'])
    for m in metrics:
        med = df[m].loc['median']
        std = df[m].loc['std']
        axs[map_[m]].hlines(y=label_metrics[m], xmin=df[m].loc['2.5% percentile'], 
                      xmax=df[m].loc['97.5% percentile'], linewidth=1, color='grey')
        axs[map_[m]].hlines(y=label_metrics[m], xmin=med-std, xmax=med+std, linewidth=3)
        axs[map_[m]].scatter(y=label_metrics[m], x=df[m].loc['median'], s=10, marker='o', color="black")
    
    fig.tight_layout()
    path = os.path.join('mimic3models/in_hospital_mortality/predictions/confvals', "overallperformance_confval_plot.jpg")   
    fig.savefig(path)

    
def calibration_overall(df, name):
    fig, axs = plt.subplots(3, 1, figsize=(6,7), sharex=True, gridspec_kw={'height_ratios': [5,1,1], 'hspace': 0})

    # PLOT 0
    axs[0].plot((0,1), (0,1), ls="--", c="#dbdbdb")
    lowess = sm.nonparametric.lowess
    z = lowess(endog=df['y_true'], exog=df['prediction'], frac=0.5, it=0, return_sorted=True)
    axs[0].plot(z[:,0], z[:,1], c='#34a84d', label='LOWESS')
    df_grouped = risk_groups(df, 10)
    axs[0].plot(df_grouped["prediction"], df_grouped["y_true"], ls=':', lw=2, marker='+', markerfacecolor='black', markeredgecolor='black', markersize=7, c='#34c0d1', label='HL groups')
    axs[0].set_ylim(0,1)

    # PLOT 1
    df_event = df[df['y_true']==1]
    axs[1].hist(df_event['prediction'], range=(0,1), color='#696969', bins=200)
    axs[1].set_yscale("log")
    axs[1].set_ylim(top=10**2)
    
    # PLOT 2
    axs[2].invert_yaxis()
    df_noevent = df[df['y_true']==0]
    axs[2].hist(df_noevent['prediction'], range=(0,1), color='#858585', bins=200)
    axs[2].set_yscale("log")
    axs[2].set_ylim(bottom=10**3)
    
    # fit the axis
    grid ='#fafafa'
    frame ='#878787'
    axs[0].legend(fontsize=14)
    plt.xlim(0,1)
    plt.xlabel("Predicted probability", fontsize=18, labelpad=10)
    axs[0].set_ylabel("Observed frequency", fontsize=16, labelpad=10)
    axs[2].set_ylabel("Counts", fontsize=16, labelpad=10)
    axs[2].yaxis.set_label_coords(-0.1,1)
    
    for ax in axs:
        plt.setp(ax.spines.values(), color=frame)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=frame)
        ax.grid(c=grid, axis='x')
        ax.tick_params(labelsize=12)
        ax.set_axisbelow(True)
        ax.axvline(x=0.5, color='darkred', linewidth=0.5, zorder=10)

    axs[0].grid(c=grid, axis='y')   
    axs[1].spines['bottom'].set_linewidth(0)
    #axs[2].spines['top'].set_linewidth(0.1) 
    axs[2].spines['top'].set_color('#c2c2c2')
    
    axs[1].text(0.88, 10, "Event", fontsize=12)
    axs[2].text(0.8, 100, "Non-event", fontsize=12)
    
    path = os.path.join('mimic3models/in_hospital_mortality/plots', name + "_calibration_overall.jpg")   
    fig.savefig(path)
    
#--------------------------- plot curves ----------------------------------------    

def plot_prc(dir_path, variable):
    variable_list = find_csv(dir_path, variable, "prc")
            
    f, ax = plt.subplots(figsize=(10, 6))
    for l in variable_list:
        path_curve = os.path.join(dir_path, l)
        df = pd.read_csv(path_curve)
        label = l.split("_")[1]
        ax.plot(df["recalls"], df["precisions"], label=label)

    plt.xlabel("Recalls")
    plt.ylabel("Precisions")
    plt.legend()
    #plt.show()
    
    
def plot_roc(dir_path, variable):
    variable_list = find_csv(dir_path, variable, "roc")
            
    f, ax = plt.subplots(figsize=(8, 5))
    for l in variable_list:
        path_curve = os.path.join(dir_path, l)
        df = pd.read_csv(path_curve)
        label = l.split("_")[1]
        ax.plot(df["fpr"], df["tpr"], label=label)

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    #plt.show()

def find_csv(dir_path, variable, curve_type):
    csv_list = os.listdir(dir_path)
    variable_list = []
    for l in csv_list:
        if (l.find(variable) != -1) and (l.find(curve_type) != -1):
            if l.find("Selfpay") == -1:
                variable_list.append(l)
    return variable_list


def side_by_side_curves(dir_path, variable):
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
    #ax2.legend()
    fig.tight_layout(pad=3.0)

    
def plot_cutoff(path):
    df = pd.read_csv(path)
    
    f, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["thresholds"], df["recalls"], color='darkred')
    ax1.set_ylabel('Recall', color='darkred', fontsize=24)
    ax1.set_xlabel('Threshold', fontsize=24)
    ax1.set_xlim(-0.02,1.02)
    ax1.set_ylim(-0.02,1.02)
    
    ax2 = ax1.twinx()
    ax2.plot(df["thresholds"], df["precisions"], color="darkslategray")
    ax2.set_ylabel('Precision', color='darkslategray', fontsize=24)
    ax2.set_xlim(-0.02,1.02)
    ax2.set_ylim(-0.02,1.02)

    hor = np.max([min(x, y) for (x, y) in zip(df['precisions'], df['recalls'])])
    ver = df['thresholds'][np.argmax([min(x, y) for (x, y) in zip(df['precisions'], df['recalls'])])]
    ax1.axhline(hor, linestyle='--', color='gray')
    ax1.axvline(ver, linestyle='--', color='gray')

    f.tight_layout()
    plt.show()
    f.savefig('mimic3models/in_hospital_mortality/plots/pr_cutoff_plot.jpg')
