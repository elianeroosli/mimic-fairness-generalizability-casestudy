from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re
import sys
from sklearn.utils import resample

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3benchmark.preprocessing import find_map_key
from mimic3models import metrics, keras_utils, common_utils 
from keras.callbacks import ModelCheckpoint, CSVLogger

#-------------------------------------- parser ----------------------------------------

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--mask_demographics', nargs='*', help='demographic variables that should be masked', default='')
# Use like: python arg.py --mask_d 1234 2345 3456 4567
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/mortality/'))
parser.add_argument('--bootstrapping',  default=False, action='store_true', help='whether to do bootstrapping')
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='mimic3models/in_hospital_mortality')
args = parser.parse_args()

if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

#----------------------------- Build and initialize readers, discretizers, normalizers -------------------------------------

train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         mask_demographics = args.mask_demographics,
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       mask_demographics = args.mask_demographics,
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

discretizer = Discretizer(mask_demographics = args.mask_demographics,
                          timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)[0]["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str_{}.start_time_zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl

#--------------------------------------- Build the model ------------------------------------------

print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)

# model name
date = utils.give_date() 
model.final_name = ".".join([date, model.say_name()]) 
print("==> model.final_name:", model.final_name)

#--------------------------------------- Compile the model -------------------------------------

print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    loss = ['binary_crossentropy'] * 2
    loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
else:
    loss = 'binary_crossentropy'
    loss_weights = None

model.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)

model.summary()

#------------------------------------ Load model weights ------------------------------------------------
n_trained_chunks = 0
if args.load_state != "":
    print(args.load_state)
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


    
#------------------------------- for TRAINING the model -------------------------------------------------------------

if args.mode == 'train':
    
    # Read training data
    train_raw_package = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
    val_raw_package = utils.load_data(val_reader, discretizer, normalizer, args.small_part)
    
    train_raw = train_raw_package["data"]
    train_dem = train_raw_package["dem"]
    val_raw = val_raw_package["data"]
    val_dem = val_raw_package["dem"]
    
    if target_repl:
        T = train_raw[0][0].shape[0]

        def extend_labels(data):
            data = list(data)
            labels = np.array(data[1])  # (B,)
            data[1] = [labels, None]
            data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
            data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
            return data

        train_raw = extend_labels(train_raw)
        val_raw = extend_labels(val_raw)

    ### Prepare metrics and storage of results
    path = os.path.join(args.output_dir, 'keras_states', date, model.final_name + '.epoch{epoch}.state')
    
    # 1) define metrics to be assessed at every epoch for train and test
    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              train_dem=train_dem,
                                                              val_dem=val_dem,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)
    # 2) make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    # 3) logs of results
    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    # fit the model: trains the model for a fixed number of epochs
    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)
    # validation_data: data on which to evaluate the loss and any model metrics at the end of each epoch
    # callbacks:
    # - metrics_callback: print the epoch results for train and validation set
    # - csv_logger: streams epoch results to a csv file
    # - saver: comes from modelcheckpoint
    

#------------------------------- for RE-TRAINING the model -------------------------------------------------------------

if args.mode == 'retrain':
    
    # Read training data
    train_raw_package = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
    val_raw_package = utils.load_data(val_reader, discretizer, normalizer, args.small_part)
    
    train_raw = train_raw_package["data"]
    train_dem = train_raw_package["dem"]
    val_raw = val_raw_package["data"]
    val_dem = val_raw_package["dem"]
    
    if target_repl:
        T = train_raw[0][0].shape[0]

        def extend_labels(data):
            data = list(data)
            labels = np.array(data[1])  # (B,)
            data[1] = [labels, None]
            data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
            data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
            return data

        train_raw = extend_labels(train_raw)
        val_raw = extend_labels(val_raw)

    ### Prepare metrics and storage of results
    path = os.path.join(args.output_dir, 'keras_states', date, model.final_name + '.epoch{epoch}.state')
    
    # 1) define metrics to be assessed at every epoch for train and test
    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              train_dem=train_dem,
                                                              val_dem=val_dem,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)
    # 2) make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    # 3) logs of results
    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    # fit the model: trains the model for a fixed number of epochs
    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)
    # validation_data: data on which to evaluate the loss and any model metrics at the end of each epoch
    # callbacks:
    # - metrics_callback: print the epoch results for train and validation set
    # - csv_logger: streams epoch results to a csv file
    # - saver: comes from modelcheckpoint
    
    
    
#-------------------------------------- for testing the model ---------------------------------------------

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader

    # load the test data
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            mask_demographics = args.mask_demographics,
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    # define our test data, labels, names and demographic info
    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]
    dems = ret["dem"]
    is_public_map = {"1": 1, "2": 1, "3": 1, "4": 0, "5": 2}
    is_public = [is_public_map[str(x)] for x in dems["Insurance"]]
    dems["binaryIns"] = is_public
    
    # model predictions and metrics
    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]

    # stratification and metric analysis
    metric_list = []
    curves_list = []
    identifier_list = []
    for dem in dems.columns:
        for val in dems[dem].unique():
            idx = dems[dem].values == val
            pred_group = predictions[dems[dem].values == val]
            label_group = [val for i,val in enumerate(labels) if idx[i]]
            m, c, i = metrics.print_metrics_binary(label_group, pred_group, dem, val)
            metric_list.append(m)
            curves_list.append(c)
            identifier_list.append(i)

    # overall metric results
    m, c, i = metrics.print_metrics_binary(labels, predictions)
    metric_list.append(m)
    curves_list.append(c)
    identifier_list.append(i)
    
    # model name
    name = model.say_name()
    filename = ".".join(["TEST", date, name])

    path_metrics = os.path.join(args.output_dir, "predictions", "metrics", filename) + ".csv"
    path_curves = os.path.join(args.output_dir, "predictions", "curves", filename)
    path_results = os.path.join(args.output_dir, "predictions", "results", filename) + ".csv"
        
    # save output
    utils.save_metrics(metric_list, path_metrics)
    utils.save_curves(curves_list, identifier_list, path_curves)
    utils.save_results(names, predictions, labels, dems, path_results)
    
    
#-------------------------------------- for testing the model ---------------------------------------------

elif args.mode == 'test_train':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader

    # load the test data
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            mask_demographics = args.mask_demographics,
                                            listfile=os.path.join(args.data, 'train_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    # define our test data, labels, names and demographic info
    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]
    dems = ret["dem"]

    # model predictions and metrics
    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    
    # stratification and metric analysis
    metric_list = []
    curves_list = []
    identifier_list = []
    for dem in dems.columns:
        for val in dems[dem].unique():
            idx = dems[dem].values == val
            pred_group = predictions[dems[dem].values == val]
            label_group = [val for i,val in enumerate(labels) if idx[i]]
            m, c, i = metrics.print_metrics_binary(label_group, pred_group, dem, val)
            metric_list.append(m)
            curves_list.append(c)
            identifier_list.append(i)
    
    # overall metric results
    m, c, i = metrics.print_metrics_binary(labels, predictions)
    metric_list.append(m)
    curves_list.append(c)
    identifier_list.append(i)
    
    # model name
    name = model.say_name()
    filename = ".".join(["TRAIN", date, name])
    
    # store metrics
    path_metrics = os.path.join(args.output_dir, "predictions", "metrics", filename) + ".csv"
    utils.save_metrics(metric_list, path_metrics)
    
    # store curves
    path_curves = os.path.join(args.output_dir, "predictions", "curves", filename)
    utils.save_curves(curves_list, identifier_list, path_curves)
    
    # store results
    path_results = os.path.join(args.output_dir, "predictions", "results", filename) + ".csv"
    utils.save_results(names, predictions, labels, dems, path_results)
    
    

else:
    raise ValueError("Wrong value for args.mode")
