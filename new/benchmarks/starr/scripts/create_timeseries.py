# +-------------------------------------------------------------------------------------------------+
# | create_timeseries.py: full STARR data processing pipeline                                       |
# |                                                                                                 |
# | Eliane Röösli (2020)                                                                            |
# +-------------------------------------------------------------------------------------------------+

import numpy as np
import pandas as pd
import argparse
import shutil
import json
import sys
import os

from benchmarks.starr.utils import create_listfile
from benchmarks.starr.preprocessing import clean_stays, clean_labs, clean_vitals, variable_list, var_map

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='build timeseries for each stay')
    parser.add_argument('-p', '--params', type=str,  help='File path of JSON parameters')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbosity of output')
    parser.parse_args()

    # parameters
    args = parser.parse_args()
    params = json.load(open(args.params))       
        
    # load tables
    stays = pd.read_csv(params['load']['stays'], sep='\t')
    labs = pd.read_csv(params['load']['labs'], sep='\t')
    vitals = pd.read_csv(params['load']['vitals'], sep='\t', low_memory=False)

    # data transformation, retrieval and cleaning
    stays = clean_stays(stays, vitals, args.verbose)
    labs = clean_labs(labs, args.verbose)
    vitals = clean_vitals(vitals, args.verbose)

    # list of variables to use
    varlist = variable_list(var_map)
    
    # clean up any old directory and create a new one (make sure no old data persists)
    ihm_path = params['load']['timeseries']
    if os.path.exists(ihm_path):
        shutil.rmtree(ihm_path)
    os.makedirs(ihm_path)
      
    if args.verbose:
        print('-'*80)
        
    # keep track of which stays get retained
    stays_retained = pd.Series([False]*stays.shape[0])

    # create individual timeseries csv file for each stay
    for idx, stay in stays.iterrows():
        
        if args.verbose:
            sys.stdout.write('\rPROCESSING ICU STAY {0} OF {1}...'.format(idx+1, stays.shape[0]))
        
        # get stay identifiers
        pat_deid = stay['pat_deid']
        stay_id = stay['stay_id']

        # get data for stay
        stay_labs = labs[(labs['pat_deid'] == pat_deid) & (labs['stay_id'] == stay_id)].sort_values('hours')
        stay_vitals = vitals[(vitals['pat_deid'] == pat_deid) & (vitals['stay_id'] == stay_id)].sort_values('hours')

        # initialize timeseries
        timeseries = pd.DataFrame(columns = varlist)
        hours = list(set(list(stay_labs['hours'])+list(stay_vitals['hours'])))
        hours.sort()
        timeseries['Hours'] = hours

        # add height and weight
        timeseries['Height'] = stay['height']
        timeseries['Weight'] = stay['weight']

        # add demographics
        timeseries['Gender'] = stay['gender']
        timeseries['Ethnicity'] = stay['race']
        timeseries['Insurance'] = stay['insurance']

        # add labs data to timeseries
        for i, lab in stay_labs.iterrows():
            timeseries.loc[timeseries['Hours'] == lab['hours'], lab['event_id']] = lab['value']

        # add vitals data to timeseries
        for i, vital in stay_vitals.iterrows():
            timeseries.loc[timeseries['Hours'] == vital['hours'], vital['event_id']] = vital['value']

        # save timeseries as csv file if at least one data point is available
        if timeseries.shape[0] > 0:
            stays_retained[idx] = True
            filename = str(pat_deid) + "_episode" + str(stay_id) + "_timeseries.csv"
            timeseries.to_csv(os.path.join(params['save']['timeseries'], filename), index=False)

    # create the listfile containing all stays
    create_listfile(params['load']['timeseries'], stays[stays_retained])
    
    # store info about retained stays
    stays_overview = stays[['pat_deid', 'stay_id']]
    stays_overview = stays_overview.assign(retained=stays_retained)
    stays_overview.to_csv(os.path.join(params['load']['timeseries'], 'retained_stays.csv'), index=False)
    
    
