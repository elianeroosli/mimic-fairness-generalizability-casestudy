# +-------------------------------------------------------------------------------------------------+
# | tools.py: functions for STARR data analysis                                                     |
# |                                                                                                 |
# | Eliane Röösli (2020)                                                                            |
# +-------------------------------------------------------------------------------------------------+

from datetime import datetime
import pandas as pd
import numpy as np
import os


def nb_years(table):
    # difference in years between two dates of table
    return (datetime.strptime(table['hosp_out'].max(), '%Y-%m-%d %H:%M:%S')
            -datetime.strptime(table['hosp_in'].min(), '%Y-%m-%d %H:%M:%S')).days/365


def time_range(labs, stays, vitals, verbose=True):
    # overview of different time ranges
    labs_dates = pd.merge(labs, stays, how='inner',
                          on=['pat_deid', 'stay_id'])[['pat_deid', 'stay_id', 'event_id', 'hosp_in', 'hosp_out']]
    vitals_dates = pd.merge(vitals, stays, how='inner', 
                            on=['pat_deid', 'stay_id'])[['pat_deid', 'stay_id', 'event_id', 'hosp_in', 'hosp_out']]
    
    if verbose:
        print("stays data from {} to {} ({:.2f} years)".format(stays['hosp_in'].min(), 
                                                               stays['hosp_out'].max(), nb_years(stays)))
        print("labs data from {} to {} ({:.2f} years)".format(labs_dates['hosp_in'].min(), 
                                                              labs_dates['hosp_out'].max(), nb_years(labs_dates)))
        print("vitals data from {} to {} ({:.2f} years)".format(vitals_dates['hosp_in'].min(), 
                                                                vitals_dates['hosp_out'].max(), nb_years(vitals_dates)))
        
        
def analyse_emptylistfile(data_path, verbose=True):
    # analyse data from empty listfiles
    empty_listfile = pd.read_csv(os.path.join(data_path, 'emptylistfile.csv'))
    empty_listfile.sort_values('hosp_in', inplace=True)
    empty_listfile['year'] = empty_listfile['hosp_in'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year)
    empty_listfile['after_date'] = empty_listfile['hosp_out'].apply(lambda x: x > '2019-07-15 12:38:00')
    
    if verbose:
        print('number of patients with no datapoints: {}'.format(empty_listfile.shape[0]))
        print('earliest hospitalization date with no data: {}'.format(empty_listfile['hosp_in'].min()))
        print('number of empty files in 2019 and 2020: {}'.format(empty_listfile[empty_listfile['year'].apply(lambda x: x in [2019, 2020])].shape[0]))
        print('number of empty files after last vital/lab event: {}'.format(empty_listfile['after_date'].sum()))
        
        
def analyse_listfile(data_path, verbose=True):
    # analyse data from empty listfiles
    listfile = pd.read_csv(os.path.join(data_path, 'listfile.csv'))
    print('average IHM:', np.round(listfile['y_true'].sum()/listfile.shape[0],4))
    