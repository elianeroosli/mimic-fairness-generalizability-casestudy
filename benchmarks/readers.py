# +-------------------------------------------------------------------------------------------------+
# | readers.py: load IHM data from timeseries csv's as input for model                              |
# |                                                                                                 |
# | Eliane Röösli (2020), adapted from Harutyunyan et al (2019)                                     |
# +-------------------------------------------------------------------------------------------------+

import os
import numpy as np
import random


class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class InHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, mask_demographics, listfile=None, period_length=48.0, data_name='mimic'):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:       Directory where timeseries files are stored.
        :param mask_demographics: List of demographic variables (strings) to be excluded from analyis
        :param listfile:          Path to a listfile. If this parameter is left `None` then
                                  `dataset_dir/listfile.csv` will be used.
        :param period_length:     Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        # data made up of name.csv files and ylabel: split it and convert label to int
        self._data_name = data_name
        self._data = [line.split(',') for line in self._data]
        if self._data_name == 'starr':
            self._data = [(x, int(y), float(c1), float(c2)) for (x, y, c1, c2) in self._data]
        else:
            self._data = [(x, int(y)) for (x, y) in self._data]
        self._mask_demographics = mask_demographics
        self._period_length = period_length
        self._demographics = ["Ethnicity", "Gender", "Insurance"]

    def _read_timeseries(self, ts_filename):
        # stores actual data used for model
        ret = []
        # stores demographic data independently of model data
        dems = []
        dems_header = []
        
        # read stays file
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            
            # determine column indices of all demographic variables and which one not to include 
            demographics_idx = []
            exclude_idx = []
            for pos, h in enumerate(header):
                if h in self._demographics:
                    demographics_idx.append(pos)
                if h in self._mask_demographics:
                    exclude_idx.append(pos)
                    
            # update headers: define demographic header and remove excluded columns from data header
            for i in sorted(demographics_idx, reverse=True):
                dems_header.append(header[i])
            for i in sorted(exclude_idx, reverse=True):
                del header[i]
            
            # read event logs    
            for line in tsfile:
                mas = line.strip().split(',')
                # store demographic data
                dem = []
                for i in sorted(demographics_idx, reverse=True):
                    dem.append(mas[i])
                # exclude demographics columns from normal data
                for i in sorted(exclude_idx, reverse=True):
                    del mas[i]
                # store the data 
                ret.append(np.array(mas))
                dems.append(np.array(dem))
                
        return (np.stack(ret), np.stack(dems), header, dems_header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        if self._data_name == 'starr':
            c1 = self._data[index][2]
            c2 = self._data[index][3]
        (X, dems, header, dems_header) = self._read_timeseries(name)

        if self._data_name == 'starr':
            data = {"X": X,
                    "t": t,
                    "y": y,
                    "c1": c1,
                    "c2": c2,
                    "header": header,
                    "name": name}
        else:
            data = {"X": X,
                    "t": t,
                    "y": y,
                    "header": header,
                    "name": name}
        
        dem_data = {"demographics": dems,
                    "header": dems_header}
        
        return  data, dem_data
                


