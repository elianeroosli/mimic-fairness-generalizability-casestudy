# +-------------------------------------------------------------------------------------------------+
# | evaluate_ihm.py: calculate bootstrapped performance metrics                                     |
# |                                                                                                 |
# | Eliane Röösli (2020), adapted from Harutyunyan et al (2019)                                     |
# +-------------------------------------------------------------------------------------------------+

import sklearn.utils as sk_utils
import numpy as np
import pandas as pd
import argparse
import simplejson as json
import os

from benchmarks.common_tools import find_map_key
from models.evaluation.metrics import print_metrics_binary
from models.ihm import utils
from models.evaluation.configs import is_public_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction', type=str)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--test_listfile', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '../../data/mimic/aug/mortality/test/listfile.csv'))
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--stratify', default=False, action='store_true')
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='models/output/mimic/predictions/confvals')
    args = parser.parse_args()

    # load the predictions file (stays, prediction, y_true, gender, insurance, ethnicity) and the listfile (stays, y_true)
    pred_df = pd.read_csv(args.prediction, index_col=False)
    test_df = pd.read_csv(args.test_listfile, index_col=False)

    # make sure the data matches
    df = test_df.merge(pred_df, left_on='stay', right_on='stay', how='left', suffixes=['_l', '_r'])
    assert (df['prediction'].isnull().sum() == 0)
    assert (df['y_true_l'].equals(df['y_true_r']))

    # drop unnecessary columns and add binary insurance data
    data = df.drop(['y_true_r', 'stay'], axis=1)
    is_public = [is_public_map[str(x)] for x in data["Insurance"]]
    data["binaryIns"] = is_public
    
    # define metrics to retain
    metrics = [('AUC of ROC', 'auroc'),
               ('AUC of PRC', 'auprc'),
               ('min(+P, Se)', 'minpse'),
               ('Prevalence', 'prev'),
               ('Accuracy', 'acc'),
               ('Precision0', 'prec0'),
               ('Precision1', 'prec1'),
               ('Recall0', 'rec0'),
               ('Recall1', 'rec1'),
               ('False Negative Rate', 'fnr'),
               ('False Positive Rate', 'fpr'),
               ('Average predicted risk', 'apr'),
               ('Average observed risk', 'prev'),
               ('Difference in average predicted and observed risk', 'diff')]

    # initialize results dict
    results = dict()
    results['n_iters'] = args.n_iters
    
    ### first store the overall results ###
    dem = "Overall"
    results[dem] = dict()
    ret, c, i = print_metrics_binary(data['y_true_l'], data['prediction'], verbose=0)           
    
    # store first value
    for (m, k) in metrics:
        results[dem][m] = dict()
        results[dem][m]['value'] = ret[k]
        results[dem][m]['runs'] = []
    
    # bootstrapping
    for i in range(args.n_iters):
        if args.stratify:
            predictions, labels = sk_utils.resample(data['prediction'], data['y_true_l'], n_samples=data.shape[0], replace=True,
                                                    stratify=data['y_true_l'])
        else: 
            predictions, labels = sk_utils.resample(data['prediction'], data['y_true_l'], n_samples=data.shape[0], replace=True)
        ret, c, i = print_metrics_binary(labels, predictions, verbose=0)
        for (m, k) in metrics:
            results[dem][m]['runs'].append(ret[k])

    # calculate statistics
    for (m, k) in metrics:
        runs = results[dem][m]['runs']
        results[dem][m]['mean'] = np.round(np.mean(runs),6)
        results[dem][m]['median'] = np.round(np.median(runs),6)
        results[dem][m]['std'] = np.round(np.std(runs),6)
        results[dem][m]['2.5% percentile'] = np.round(np.percentile(runs, 2.5),6)
        results[dem][m]['97.5% percentile'] = np.round(np.percentile(runs, 97.5),6)
        del results[dem][m]['runs']
    
    
    ### repeat same procedure for demographic groups data ###
    for dem in data[['binaryIns', 'Insurance', 'Gender', 'Ethnicity']].columns:
        results[dem] = dict()
        
        sorted_values = sorted(data[dem].unique())
        for val in sorted_values:
            dem_val = find_map_key(str(dem), int(val))
            results[dem][dem_val] = dict()
            
            # get data associated to demographic group
            idx = data[dem].values == val
            pred_group = data[data[dem].values == val]['prediction']
            label_group = [val for i,val in enumerate(data['y_true_l']) if idx[i]]
            
            ret, c, i = print_metrics_binary(label_group, pred_group, dem, val, verbose=0)

            # store first value
            for (m, k) in metrics:
                results[dem][dem_val][m] = dict()
                results[dem][dem_val][m]['value'] = ret[k]
                results[dem][dem_val][m]['runs'] = []

            # bootstrapping
            for i in range(args.n_iters):
                if args.stratify:
                    pred_group_, label_group_ = sk_utils.resample(pred_group, label_group, n_samples=len(pred_group), replace=True, stratify=label_group)
                else:
                    pred_group_, label_group_ = sk_utils.resample(pred_group, label_group, n_samples=len(pred_group), replace=True)
                ret, c, i = print_metrics_binary(label_group_, pred_group_, verbose=0)
                for (m, k) in metrics:
                    results[dem][dem_val][m]['runs'].append(ret[k])

            # calculate statistics
            for (m, k) in metrics:
                runs = results[dem][dem_val][m]['runs']
                results[dem][dem_val][m]['mean'] = np.round(np.mean(runs),6)
                results[dem][dem_val][m]['median'] = np.round(np.median(runs),6)
                results[dem][dem_val][m]['std'] = np.round(np.std(runs),6)
                results[dem][dem_val][m]['2.5% percentile'] = np.round(np.percentile(runs, 2.5),6)
                results[dem][dem_val][m]['97.5% percentile'] = np.round(np.percentile(runs, 97.5),6)
                del results[dem][dem_val][m]['runs']

   
    # save the results      
    date = utils.give_date()
    if args.stratify:
        name = ".".join([date, "stratified_ihm_results.json"]) 
    else:
        name = ".".join([date, "nonstratified_ihm_results.json"]) 
    
    path = os.path.join(args.output_dir, name)
    print("Saving the results in {} ...".format(path))
    with open(path, 'w') as f:
        json.dump(results, f, ignore_nan=True)

    if args.verbose:
        print(results)


if __name__ == "__main__":
    main()
