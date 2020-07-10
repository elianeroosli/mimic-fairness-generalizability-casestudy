# +-------------------------------------------------------------------------------------------------+
# | metrics.py: calculate all metrics for a binary classification task                              |
# |                                                                                                 |
# | Eliane Röösli (2020), adapted from Harutyunyan et al (2019)                                     |
# +-------------------------------------------------------------------------------------------------+

import numpy as np
from sklearn import metrics

from benchmarks.common_tools import find_map_key


#### decompensation, in-hospital mortality #### 

def print_metrics_binary(y_true, predictions, dem_variable=None, dem_value=None, verbose=1):
    predictions = np.array(predictions)
    
    # if risk prediction only made for one of the two outcomes
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    # predictions are done with a 50% cutoff: label = 1 if risk score is >= 50%
    # cf[i][j]: number of observations known to be in group i and predicted to be in group j
    # e.g. FP: cf[0][1]
    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1), labels = [0, 1])
    if verbose:
        print("\nconfusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    # a) metrics from confusion matrix
    acc = np.round(float((cf[0][0] + cf[1][1]) / np.sum(cf)),6)
    prec0 = np.round(float(cf[0][0] / (cf[0][0] + cf[1][0])),6)
    prec1 = np.round(float(cf[1][1] / (cf[1][1] + cf[0][1])),6)
    rec0 = np.round(float(cf[0][0] / (cf[0][0] + cf[0][1])),6)
    rec1 = np.round(float(cf[1][1] / (cf[1][1] + cf[1][0])),6)
    fnr = np.round(float(cf[1][0] / (cf[1][0] + cf[1][1])),6)
    fpr = np.round(float(cf[0][1] / (cf[0][1] + cf[0][0])),6)
    ppr = np.round(float((cf[1][1] + cf[0][1])/np.sum(cf)),6)
    
    # b) ROC
    try:
        auroc = np.round(metrics.roc_auc_score(y_true, predictions[:, 1]),6)
    except:
        auroc = np.nan
    (roc_fpr, roc_tpr, roc_thresholds) = metrics.roc_curve(y_true, predictions[:, 1])
    
    # c) PRC
    (prc_precisions, prc_recalls, prc_thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = np.round(metrics.auc(prc_recalls, prc_precisions),6)
    # this is necessary as vector is by sklearn implementation one element shorter than prec & rec
    prc_thresholds = np.append(prc_thresholds,1) 
    
    # d) best threshold
    minpse = np.round(np.max([min(x, y) for (x, y) in zip(prc_precisions, prc_recalls)]),6)
    
    # e) average risks
    apr = np.round(np.mean(predictions[:, 1]),6)
    nb = len(y_true)
    prev = np.round(sum(y_true)/nb,6)
    diff = apr - prev
    identifier = "overall_overall"

    if verbose:
        if dem_variable != None:        
            print("\nstratified metric for:")
            print("demographic variable:", dem_variable)
            val = find_map_key(str(dem_variable), int(dem_value))
            print("value:", val)
            identifier = dem_variable + "_" + val.replace(" ", "")
            print("\noverall metrics:")
    
        print("prevalence of IHM = {}".format(prev))
        print("average predicted risk of IHM = {}".format(apr))
        print("difference in average predicted and observed risk = {}".format(diff))
        print("number samples = {}".format(nb))
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("false negative rate = {}".format(fnr))
        print("false positive rate = {}".format(fpr))
        print("positive prediction rate = {}".format(ppr))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))
        print("\n-------------------------------")
        
    metric_summary = {"dem_var": dem_variable,
            "dem_val": dem_value,
            "prev": prev,
            "apr": apr,
            "diff": diff,
            "nb": nb,
            "acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "fnr": fnr,
            "fpr": fpr,
            "ppr": ppr,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse}
    
    # save arrays defining the roc and prc curves
    roc = {"fpr": roc_fpr,
            "tpr": roc_tpr, 
            "thresholds": roc_thresholds}
    prc = {"precisions": prc_precisions,
            "recalls": prc_recalls,
            "thresholds": prc_thresholds}
    curves = [roc, prc]
    
    # get the actual name for the demographic value
    if dem_variable != None:
        metric_summary["dem_val"] = find_map_key(str(dem_variable), int(dem_value))
    
    return metric_summary, curves, identifier

