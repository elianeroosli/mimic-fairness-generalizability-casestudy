from __future__ import absolute_import
from __future__ import print_function

from mimic3models.metrics import print_metrics_binary
from mimic3benchmark.preprocessing import find_map_key
import sklearn.utils as sk_utils
import numpy as np
import pandas as pd
import argparse
import json
import os

    
def first_values(results_dict, ret, metrics):     
    for (m, k) in metrics:
        results_dict[m] = dict()
        results_dict[m]['value'] = ret[k]
        results_dict[m]['runs'] = []
    return results_dict
        
def run_stats(results_dict, metrics):
    for (m, k) in metrics:
        runs = results_dict[m]['runs']
        results_dict[m]['mean'] = np.round(np.mean(runs),6)
        results_dict[m]['median'] = np.round(np.median(runs),6)
        results_dict[m]['std'] = np.round(np.std(runs),6)
        results_dict[m]['2.5% percentile'] = np.round(np.percentile(runs, 2.5),6)
        results_dict[m]['97.5% percentile'] = np.round(np.percentile(runs, 97.5),6)
        del results_dict[m]['runs']
    return results_dict
    
    
    
    """ dem = "Overall"
    results[dem] = dict()
    ret, c, i = print_metrics_binary(data['y_true_l'], data['prediction'], verbose=0)           
    for (m, k) in metrics:
        results[dem][m] = dict()
        results[dem][m]['value'] = ret[k]
        results[dem][m]['runs'] = []
        
    
    for i in range(args.n_iters):
                pred_group_, label_group_ = sk_utils.resample(pred_group, label_group, n_samples=len(pred_group), replace=True, stratify=label_group)
                ret, c, i = print_metrics_binary(label_group_, pred_group_, verbose=0)
                for (m, k) in metrics:
                    results[dem][dem_val][m]['runs'].append(ret[k])

            for (m, k) in metrics:
                runs = results[dem][dem_val][m]['runs']
                results[dem][dem_val][m]['mean'] = np.round(np.mean(runs),6)
                results[dem][dem_val][m]['median'] = np.round(np.median(runs),6)
                results[dem][dem_val][m]['std'] = np.round(np.std(runs),6)
                results[dem][dem_val][m]['2.5% percentile'] = np.round(np.percentile(runs, 2.5),6)
                results[dem][dem_val][m]['97.5% percentile'] = np.round(np.percentile(runs, 97.5),6)
                del results[dem][dem_val][m]['runs']"""