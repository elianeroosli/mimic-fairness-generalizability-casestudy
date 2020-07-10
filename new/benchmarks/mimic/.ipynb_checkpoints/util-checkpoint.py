# +-------------------------------------------------------------------------------------------------+
# | util.py: load dataframe from csv                                                                |
# |                                                                                                 |
# | Eliane Röösli (2020), adapted from Harutyunyan et al (2019)                                     |
# +-------------------------------------------------------------------------------------------------+

import pandas as pd
import csv


def dataframe_from_csv(path, header=0, index_col=0):
    with open(path, "r") as f:
        d_reader = csv.DictReader(f)
    return pd.read_csv(path, header=header, index_col=index_col)
