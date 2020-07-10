# +-------------------------------------------------------------------------------------------------+
# | extract_episodes_from_subjects.py: create timeseries                                            |
# |                                                                                                 |
# | Eliane Röösli (2020), adapted from Harutyunyan et al (2019)                                     |
# +-------------------------------------------------------------------------------------------------+

import os
import sys
import numpy as np
import pandas as pd
import argparse

from benchmarks.mimic.subject import read_stays, read_diagnoses, read_events, get_events_for_stay, add_hours_elapsed_to_events
from benchmarks.mimic.subject import convert_events_to_timeseries, get_first_valid_from_timeseries
from benchmarks.mimic.preprocessing import read_itemid_to_variable_map, map_itemids_to_variables, clean_events
from benchmarks.mimic.preprocessing import assemble_episodic_data

def is_subject_folder(x):
    return str.isdigit(x)

parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--add_demographics', action='store_true', help='add demographical variables')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/itemid_to_variable_map.csv'),
                    help='CSV containing ITEMID-to-VARIABLE map.')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for VARIABLEs.')
parser.add_argument('--verbose', '-v', type=int, help='Level of verbosity in output.', default=1)
args, _ = parser.parse_known_args()

var_map = read_itemid_to_variable_map(args.variable_map_file)
variables = list(var_map.VARIABLE.unique())

subdirectories = os.listdir(args.subjects_root_path)
subjects = list(filter(is_subject_folder, subdirectories))
nb_patients = len(subjects)

reading_error = 0
noevents_error = 0
nodata_error = 0
problems_patients = 0
problems_stays = 0

#--------------------------------- for each subject ------------------------------------------------------------------

for (index, subject_dir) in enumerate(subjects):
    sys.stdout.write('\rSUBJECT {0} of {1}...'.format(index+1, nb_patients))
    
    # get subject_id from directory name
    dn = os.path.join(args.subjects_root_path, subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue

    # get the stays, diagnoses and events data for each subject
    try:
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events = read_events(os.path.join(args.subjects_root_path, subject_dir))
    except:
        sys.stdout.write('error reading from disk!')
        problems_patients += 1
        reading_error += 1
        continue

    # assemble the data associated with the episode
    episodic_data = assemble_episodic_data(stays, diagnoses)

    # --------------------------------------- events data -------------------------------------------
    
    # mapping and cleaning of events
    events = map_itemids_to_variables(events, var_map)
    events = clean_events(events)
    if events.shape[0] == 0:
        sys.stdout.write('no valid events!')
        problems_patients += 1
        noevents_error += 1
        continue
    
    # convert event logs into a timeseries 
    timeseries = convert_events_to_timeseries(events, variables=variables)
    
    # NEW preparation: add columns for demographics
    if args.add_demographics:
        demographic_vars = ['Ethnicity', 'Gender', 'Insurance']
        for var in demographic_vars:
            timeseries[var] = 0

    # for each stay of a given subject, extract the events during an episode (stay at icu)
    for i in range(stays.shape[0]):
        stay_id = stays.ICUSTAY_ID.iloc[i]
        
        # start and endtime of episode (single ICU stay)
        intime = stays.INTIME.iloc[i]
        outtime = stays.OUTTIME.iloc[i]
        
        # get the events for each individual episode
        episode = get_events_for_stay(timeseries.copy(), stay_id, intime, outtime)
        if episode.shape[0] == 0:
            sys.stdout.write(' (no data!)')
            sys.stdout.flush()
            problems_stays += 1
            nodata_error += 1
            continue
            
        # NEW: add demographics
        if args.add_demographics:
            for var in demographic_vars:
                episode[var] = episodic_data[var].iloc[0]

        # add time to all events
        episode = add_hours_elapsed_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)
        
        # save general episodic data
        episodic_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode, 'Weight')
        episodic_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode, 'Height')
        episodic_data.loc[episodic_data.index==stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                            'episode{}.csv'.format(i+1)), index_label='Icustay')
        # save events data of episode
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i+1)),
                       index_label='Hours')

print('\nproblems with patients:', problems_patients)
print('problems with stays:', problems_stays)
print('reading error:', reading_error)
print('no valid events:', noevents_error)
print('no data:', nodata_error)