from __future__ import absolute_import
from __future__ import print_function

from models.mimic import common_utils
import numpy as np
import pandas as pd
import os
import datetime
import pytz


def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    # number of samples to load is by default the total number of samples available
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    # store the data in ret
    ret, dems = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    # get demographics data
    dem = demographics(dems["demographics"], dems["header"])
    
    # data gets discretized and normalized
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), labels)
    # data: list (stays) of list (time steps: 48) of list (variables)
    if not return_names:
        return {"data": whole_data, "dem": dem}
    return {"data": whole_data, "dem": dem, "names": names}


def give_date():
    # returns string of current date and time (hour & minutes) for the PST timezone
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    pst_now = utc_now.astimezone(pytz.timezone("America/Los_Angeles"))
    return str(pst_now.date()) + "_" + str(pst_now.hour) + "-" + str(pst_now.minute)



def demographics(data, header):
    # extract and store summary demographics
    summary_demographics = []
    for stays in range(len(data)):
        stay_demographics = []
        for ch in range(len(data[stays][0])):    
            stay_demographics.append(data[stays][0][ch])
        summary_demographics.append(stay_demographics)
    
    return pd.DataFrame(summary_demographics, columns=header)



def save_results(names, pred, y_true, dems, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true,Insurance,Gender,Ethnicity\n")
        for (name, x, y, dem) in zip(names, pred, y_true, dems.values.tolist()):
            f.write("{},{:.6f},{},{},{},{}\n".format(name, x, y, dem[0], dem[1], dem[2]))

            
def save_metrics(metric_list, path):
    common_utils.create_directory(os.path.dirname(path))     
    with open(path, 'w') as f:
        header = ",".join(metric_list[0].keys())     
        f.write(header+"\n")
        for metric in metric_list:
            str_values = [str(val) for val in metric.values()]
            values = ",".join(str_values)
            f.write(values+ "\n")       
            
            
def save_curves(curves_list, identifier_list, path_dir):
    common_utils.create_directory(path_dir)
    types = ["roc", "prc"]
    for i, curve_pairs in enumerate(curves_list):
        for j, curve_dict in enumerate(curve_pairs):
            path_curve = os.path.join(path_dir, identifier_list[i] + "_" + types[j]) + ".csv"
            df = pd.DataFrame.from_dict(curve_dict) 
            df.to_csv(path_curve, header=df.columns, index=False)


