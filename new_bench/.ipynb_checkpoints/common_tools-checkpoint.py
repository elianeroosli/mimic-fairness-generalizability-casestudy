import os
import numpy as np

from benchmarks.mimic.readers import InHospitalMortalityReader
from models.mimic.preprocessing import Discretizer
from models.mimic.in_hospital_mortality import utils
from models.mimic import common_utils


#----------------------------- DATA ANALYSIS TOOLS: IMPUTATION RATES AFTER DISCRETIZING INTO 1H BINS --------------------

def discretize_data(data_path, mask_demographics, source):
    
    # initialize data readers for given source
    if source == 'mimic':
        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'train'),
                                         mask_demographics = mask_demographics,
                                         listfile=os.path.join(data_path, 'train_listfile.csv'),
                                         period_length=48.0)

        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'train'),
                                               mask_demographics = mask_demographics,
                                               listfile=os.path.join(data_path, 'val_listfile.csv'),
                                               period_length=48.0)

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'test'),
                                               mask_demographics = mask_demographics,
                                               listfile=os.path.join(data_path, 'test_listfile.csv'),
                                               period_length=48.0)
        
        readers = [train_reader, val_reader, test_reader]
        
    elif source == 'starr':
        reader = InHospitalMortalityReader(dataset_dir=data_path,
                                         mask_demographics = mask_demographics,
                                         listfile=os.path.join(data_path, 'listfile.csv'),
                                         period_length=48.0)
        
        readers = [reader]
        
    else:
        print('source not valid')
        
    # initialize discretizer    
    discretizer = Discretizer(mask_demographics = mask_demographics,
                          timestep=1.0,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')
    
    # read and discretize data
    for reader in readers:    
        N = reader.get_number_of_examples()
        ret, dems = common_utils.read_chunk(reader, N)
        data = ret["X"]
        ts = ret["t"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        
    return discretizer


def print_stats(dst):
    print("statistics of discretizer:")
    print("\tconverted {} examples".format(dst._done_count))
    print("\taverage unused data = {:.2f}%".format(100.0 * dst._unused_data_sum / dst._done_count))
    print("\taverage completely empty bins = {:.2f}%".format(100.0 * dst._empty_bins_sum / dst._done_count))
    print("\t------------------------------")
    for i in range(len(dst._id_to_channel)):
        print("\t{}:".format(dst._id_to_channel[i]))
        print("\t  mean = {:.1f}/48 ({:.1f}%)".format(np.mean(dst._nonimputed_channels[i]), 100*np.mean(dst._nonimputed_channels[i])/48))
        print("\t  proportion of ICU stays with:")
        print("\t    no data = {:.1f}%".format(100*sum([j==0 for j in dst._nonimputed_channels[i]])/dst._done_count))
        print("\t    all data = {:.1f}%".format(100*sum([j==48 for j in dst._nonimputed_channels[i]])/dst._done_count))
        
        
def print_stats_report(dst):
    for i in range(len(dst._id_to_channel)):
        print("{:.2f}%".format(100*sum([j==0 for j in dst._nonimputed_channels[i]])/dst._done_count))
    print("-"*40)
    for i in range(len(self._id_to_channel)):  
        print("{:.2f}%".format(100*sum([j==48 for j in dst._nonimputed_channels[i]])/dst._done_count))
    print("-"*40)
    for i in range(len(self._id_to_channel)):  
        print("{:.1f} ({:.1f}%)".format(np.mean(dst._nonimputed_channels[i]), 100*np.mean(dst._nonimputed_channels[i])/48))