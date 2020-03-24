from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import pandas as pd

from mimic3benchmark.util import *


def read_stays(subject_path):
    stays = dataframe_from_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.DOB = pd.to_datetime(stays.DOB)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays


def read_diagnoses(subject_path):
    return dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)


def read_events(subject_path, remove_null=True):
    events = dataframe_from_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events.loc[events.VALUE.notnull()]
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    events.HADM_ID = events.HADM_ID.fillna(value=-1).astype(int)
    events.ICUSTAY_ID = events.ICUSTAY_ID.fillna(value=-1).astype(int)
    events.VALUEUOM = events.VALUEUOM.fillna('').astype(str)
    # events.sort_values(by=['CHARTTIME', 'ITEMID', 'ICUSTAY_ID'], inplace=True)
    return events


def get_events_for_stay(events, icustayid, intime=None, outtime=None):
    idx = (events.ICUSTAY_ID == icustayid)
    if intime is not None and outtime is not None:
        idx = idx | ((events.CHARTTIME >= intime) & (events.CHARTTIME <= outtime))
    events = events.loc[idx]
    del events['ICUSTAY_ID']
    return events


def add_hours_elapsed_to_events(events, dt, remove_charttime=True):
    events.loc[:,'HOURS'] = (events.CHARTTIME - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
    if remove_charttime:
        del events['CHARTTIME']
    return events


def add_ethnicity_to_events(events, episodic_data, variables):
    # create new events entry for ethnicity: important is VALUE and VARIABLE
    demographics = {'SUBJECT_ID': events.SUBJECT_ID.iloc[0], 'HADM_ID': events.HADM_ID.iloc[0], 'ICUSTAY_ID': events.ICUSTAY_ID.iloc[0],
                    'CHARTTIME': events.CHARTTIME.iloc[0], 'ITEMID': np.nan, 'VALUE': episodic_data['Ethnicity'].iloc[0], 'VALUEUOM': np.nan, 
                    'VARIABLE': 'Ethnicity', 'MIMIC_LABEL': np.nan}
    variables.append('Ethnicity')
    return events.append(demographics, ignore_index=True), variables


def add_gender_to_events(events, episodic_data, variables):
    # create new events entry for gender: important is VALUE and VARIABLE
    demographics = {'SUBJECT_ID': events.SUBJECT_ID.iloc[0], 'HADM_ID': events.HADM_ID.iloc[0], 'ICUSTAY_ID': events.ICUSTAY_ID.iloc[0],
                    'CHARTTIME': events.CHARTTIME.iloc[0], 'ITEMID': np.nan, 'VALUE': episodic_data['Gender'].iloc[0], 'VALUEUOM': np.nan, 
                    'VARIABLE': 'Gender', 'MIMIC_LABEL': np.nan}
    variables.append('Gender')
    return events.append(demographics, ignore_index=True), variables


def convert_events_to_timeseries(events, episodic_data, variable_column='VARIABLE', variables=[]):
    # add demographic variables to events
    #events, variables = add_gender_to_events(events, episodic_data, variables)
    #events, variables = add_ethnicity_to_events(events, episodic_data, variables)
    
    # get metadata of events: charttime and icustay  
    metadata = events[['CHARTTIME', 'ICUSTAY_ID']].sort_values(by=['CHARTTIME', 'ICUSTAY_ID'])\
                    .drop_duplicates(keep='first').set_index('CHARTTIME')
    
    # get the charttime, variable name and value and order it by charttime
    timeseries = events[['CHARTTIME', variable_column, 'VALUE']]\
                    .sort_values(by=['CHARTTIME', variable_column, 'VALUE'], axis=0)\
                    .drop_duplicates(subset=['CHARTTIME', variable_column], keep='last')

    # create a df with x: CHARTTIME, y: the entries of variable_column (contains the names of our variables); and entries filled up with the value
    timeseries = timeseries.pivot(index='CHARTTIME', columns=variable_column, values='VALUE')
    
    # add info about icustay_id
    timeseries = timeseries.merge(metadata, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    
    #print(timeseries)
    
    # if a variable is completely missing in the timeseries, add a column with nans
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries


def get_first_valid_from_timeseries(timeseries, variable):
    if variable in timeseries:
        idx = timeseries[variable].notnull().copy()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan
