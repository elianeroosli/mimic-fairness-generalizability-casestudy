# +-------------------------------------------------------------------------------------------------+
# | common_tools.py: shared functions among mimic and starr                                         |
# |                                                                                                 |
# | Eliane Röösli (2020)                                                                            |
# +-------------------------------------------------------------------------------------------------+

import os
import numpy as np

from benchmarks.readers import InHospitalMortalityReader
from models.preprocessing import Discretizer
from models.ihm import utils
from models import common_utils


#----------------------------------- DEMOGRAPHIC VARIABLE MAPS --------------------------------

# return the key for a given value in the map associated with a given demographic variable
def find_map_key(dem, val_search):
    if (dem == 'Gender') or (dem == 'gender'):
        return get_key(g_map, val_search)
    if (dem == 'Ethnicity') or (dem == 'ethnicity'):
        return get_key(e_map, val_search)
    if (dem == 'Insurance') or (dem == 'insurance'):
        return get_key(i_map, val_search)
    if (dem == 'binaryIns'):
        return get_key(ispublic_map, val_search)

# get the key given a dictionary and a value
def get_key(dictionary, val_search):
    for key, val in dictionary.items():  
        if val == val_search:
            return key

# gender map
g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}


def transform_gender(gender_series):
    global g_map    # to access the variable defined outside of the function
    return { 'Gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER']) }

# ethnicity map: there are many more subtypes but they all contain one of these keywords at first position
e_map = {'ASIAN': 1,
         'BLACK': 2,
         'CARIBBEAN ISLAND': 2,
         'HISPANIC': 3,
         'SOUTH AMERICAN': 3,
         'WHITE': 4,
         'MIDDLE EASTERN': 4,
         'PORTUGUESE': 4,
         'OTHER': 0,
         'AMERICAN INDIAN': 0,
         'NATIVE HAWAIIAN': 0,
         'UNABLE TO OBTAIN': 0,
         'PATIENT DECLINED TO ANSWER': 0,
         'UNKNOWN': 0,
         '': 0}

def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        # replace(old, new); split(separator) returns list of strings: only take first one
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {'Ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}

ispublic_map = {'Public':1,
         'Private': 0,
         'Selfpay': 2,
        }

i_map = {'Government':1,
         'Medicare': 2,
         'Medicaid': 3,
         'Private': 4,
         'Self Pay': 5,
         'Other': 0,
         '': 0}

def transform_insurance(insurance_series):
    global i_map
    return {'Insurance': insurance_series.fillna('').apply(lambda s: i_map[s] if s in i_map else i_map['Other']) }



#----------------------------- DATA ANALYSIS TOOLS: IMPUTATION RATES AFTER DISCRETIZING INTO 1H BINS --------------------

def discretize_data(data_path, mask_demographics):
    
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
                  
    # initialize discretizer    
    discretizer = Discretizer(mask_demographics = mask_demographics,
                          timestep=1.0,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')
    
    # read and discretize data
    for reader in [train_reader, val_reader, test_reader]:    
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
        print("{:.1f}%".format(100*sum([j==0 for j in dst._nonimputed_channels[i]])/dst._done_count))
    print("-"*40)
    for i in range(len(dst._id_to_channel)):  
        print("{:.1f}%".format(100*sum([j==48 for j in dst._nonimputed_channels[i]])/dst._done_count))
    print("-"*40)
    for i in range(len(dst._id_to_channel)):  
        print("{:.1f} ({:.1f})".format(np.mean(dst._nonimputed_channels[i]), 100*np.mean(dst._nonimputed_channels[i])/48))