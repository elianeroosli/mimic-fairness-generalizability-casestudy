from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import platform
import pickle
import json
import os

### processing functions to adjust the demographic channels included
def process_conf(config, mask_demographics):
    config['id_to_channel'] = process_conf_list(config['id_to_channel'], mask_demographics)
    config['is_categorical_channel'] = process_conf_dict(config['is_categorical_channel'], mask_demographics)
    config['possible_values'] = process_conf_dict(config['possible_values'], mask_demographics)
    config['normal_values'] = process_conf_dict(config['normal_values'], mask_demographics)
    return config

def process_conf_dict(conf_dict, mask_demographics):
    exclude_key = []
    for key, val in conf_dict.items():
        if key in mask_demographics:
            exclude_key.append(key)
    for key in exclude_key:
        del conf_dict[key]
    return conf_dict

def process_conf_list(conf_list, mask_demographics):
    exclude_idx = []
    for pos,ch in enumerate(conf_list):
        if ch in mask_demographics:
            exclude_idx.append(pos)
    for i in sorted(exclude_idx, reverse=True):
        del conf_list[i]
    return conf_list



class Discretizer:
    def __init__(self, mask_demographics, timestep=0.8, store_masks=True, impute_strategy='zero', start_time='zero',
                 config_path=os.path.join(os.path.dirname(__file__), 'resources/discretizer_config.json')):
        """ This class can be used for re-sampling time-series data into regularly spaced intervals
            and for imputing the missing values.
  
        :param timestep: Defines the length of intervals.
        :param store_masks: When this parameter is True, the discretizer will append a binary vector to
                            the data of each time-step. This binary vector specifies which entries are imputed.
        :param impute_strategy: Specifies the imputation strategy. Possible values are 'zero', 'normal_value',
                               'previous' and 'next'.
        :param start_time: Specifies when to start to re-sample the data. Possible values are 'zero' and 'relative'.
                           In case of 'zero' the discretizer will start to re-sample the data from time 0 and in case of 
                           'relative it will start to re-sample from the moment when the first ICU event happens'.
        """
        with open(config_path) as f:
            config = json.load(f)
            config = process_conf(config, mask_demographics)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy

        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0
        self._nonimputed_channels = [[] for i in range(len(self._id_to_channel))]

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]

        ### to keep track of the data discretizing process
        data = np.zeros(shape=(N_bins, cur_len), dtype=float) # store the data (default imputation: zero)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int) # stores which (bin_id, channel_id) pairs have been used in data
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0 # total number of values stored in data
        unused_data = 0 # number of values not used as an observation for (bin_id, channel_id) already exists

        ### stores value of channel in the given time bin to data 
        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            # one-hot encoding for categorical channels
            if self._is_categorical_channel[channel]:
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                if value == 'None' or value == '>500' or value == '>600' or value == '<20' or value=='<6.50':
                    value = np.nan
                data[bin_id, begin_pos[channel_id]] = float(value)

                
        ### for all timesteps
        for row in X:
            # assign sample to time bin
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            ### for every value in a given row
            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                # update summary statistics
                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1

                # write value row[j] to data
                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

                
        ### impute missing values: by default, use zero imputation (initialization of data variable)
        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        ### 1) impute previous or normal value
        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    # if there is already a value stored for the (bin_id, channel_id) pair: store it and continue
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    # if there is no value:
                    # 1) impute normal value
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    # 2) impute previous value if it exists or otherwise the normal value
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    # store imputed value in data
                    write(data, bin_id, channel, imputed_value, begin_pos)

        ### 2) impute next value: same structure as above but iterate over bin_id's in reverse order
        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            # iterate over bin_id's in reverse order
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        # count the number of (bin,channel) pairs that had to be imputed (have a zero in the mask)
        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)
        
        for i in range(N_channels):
            self._nonimputed_channels[i].append(np.sum(mask[:, i]))

        # add mask as features to the data (as last columns)
        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)

    def print_statistics(self):
        print("statistics of discretizer:")
        print("\tconverted {} examples".format(self._done_count))
        print("\taverage unused data = {:.2f}%".format(100.0 * self._unused_data_sum / self._done_count))
        print("\taverage empty bins = {:.2f}%".format(100.0 * self._empty_bins_sum / self._done_count))
        for i in range(len(self._id_to_channel)):
            print("\tmean impute rate for {} = {:.2f}%".format(self._id_to_channel[i], 100.0 * self._imputed_channels_sum[i] / self._done_count))
            print("\tspread of impute rate for {}:".format(self._id_to_channel[i]))
            print("\t  min = {:.2f}%".format(min(100*self._imputed_channels_indiv[i])))
            print("\t  mean= {:.2f}%".format(np.mean(100*self._imputed_channels_indiv[i])))
            print("\t  max = {:.2f}%".format(max(100*self._imputed_channels_indiv[i])))


class Normalizer:
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0

    def _feed_data(self, x):
        x = np.array(x)
        self._count += x.shape[0]
        if self._sum_x is None:
            self._sum_x = np.sum(x, axis=0)
            self._sum_sq_x = np.sum(x**2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x**2, axis=0)

    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(1.0/(N - 1) * (self._sum_sq_x - 2.0 * self._sum_x * self._means + N * self._means**2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj={'means': self._means,
                             'stds': self._stds},
                        file=save_file,
                        protocol=2)

    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding='latin1')
            self._means = dct['means']
            self._stds = dct['stds']

    def transform(self, X):
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret
