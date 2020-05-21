from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re
import sys

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3benchmark.preprocessing import find_map_key
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models.in_hospital_mortality import utils
from mimic3models.evaluation import metrics
from mimic3models import keras_utils, common_utils
from keras.callbacks import ModelCheckpoint, CSVLogger

#-------------------------------------- parser ----------------------------------------

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--mask_demographics', nargs='*', help='demographic variables that should be masked', default='')
# Use like: python arg.py --mask_d 1234 2345 3456 4567
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='mimic3models/in_hospital_mortality')
args = parser.parse_args()


#----------------------------- Build and initialize readers, discretizers, normalizers -------------------------------------

train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         mask_demographics = args.mask_demographics,
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       mask_demographics = args.mask_demographics,
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                       mask_demographics = args.mask_demographics,
                                       listfile=os.path.join(args.data, 'test_listfile.csv'),
                                       period_length=48.0)

discretizer = Discretizer(mask_demographics = args.mask_demographics,
                          timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

#discretizer_header = discretizer.transform(train_reader.read_example(0)[0]["X"])[1].split(',')
#print(discretizer_header)

for reader in [train_reader, val_reader, test_reader]:    
    N = reader.get_number_of_examples()
    ret, dems = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
discretizer.print_statistics()    
