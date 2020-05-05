from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import csv


def dataframe_from_csv(path, header=0, index_col=0):
    with open(path, "r") as f:
        d_reader = csv.DictReader(f)
        #print(d_reader.fieldnames)
    return pd.read_csv(path, header=header, index_col=index_col)
