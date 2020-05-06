from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from sklearn import metrics
from mimic3benchmark.preprocessing import find_map_key


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


#### phenotyping ####

def print_metrics_multilabel(y_true, predictions, verbose=1):
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return {"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted}


#### Length-of-stay prediction ####

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100


def print_metrics_regression(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    predictions = np.maximum(predictions, 0).flatten()
    y_true = np.array(y_true)

    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Custom bins confusion matrix:")
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins,
                                      weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    if verbose:
        print("Mean absolute deviation (MAD) = {}".format(mad))
        print("Mean squared error (MSE) = {}".format(mse))
        print("Mean absolute percentage error (MAPE) = {}".format(mape))
        print("Cohen kappa score = {}".format(kappa))

    return {"mad": mad,
            "mse": mse,
            "mape": mape,
            "kappa": kappa}


class LogBins:
    nbins = 10
    means = [0.611848, 2.587614, 6.977417, 16.465430, 37.053745,
             81.816438, 182.303159, 393.334856, 810.964040, 1715.702848]


def get_bin_log(x, nbins, one_hot=False):
    binid = int(np.log(x + 1) / 8.0 * nbins)
    if binid < 0:
        binid = 0
    if binid >= nbins:
        binid = nbins - 1

    if one_hot:
        ret = np.zeros((LogBins.nbins,))
        ret[binid] = 1
        return ret
    return binid


def get_estimate_log(prediction, nbins):
    bin_id = np.argmax(prediction)
    return LogBins.means[bin_id]


def print_metrics_log_bins(y_true, predictions, verbose=1):
    y_true_bins = [get_bin_log(x, LogBins.nbins) for x in y_true]
    prediction_bins = [get_bin_log(x, LogBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("LogBins confusion matrix:")
        print(cf)
    return print_metrics_regression(y_true, predictions, verbose)


# for length of stay prediction: 10 bins
class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)
    means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
             131.579534, 155.643957, 179.660558, 254.306624, 585.325890]


# find th corresponding bin for x (in hours)
def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None


def get_estimate_custom(prediction, nbins):
    bin_id = np.argmax(prediction)
    assert 0 <= bin_id < nbins
    return CustomBins.means[bin_id]


def print_metrics_custom_bins(y_true, predictions, verbose=1):
    return print_metrics_regression(y_true, predictions, verbose)
