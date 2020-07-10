# +-------------------------------------------------------------------------------------------------+
# | mimic3csv.py: read and process (filter, break up) mimic csv tables                              |
# |                                                                                                 |
# | Eliane Röösli (2020), adapted from Harutyunyan et al (2019)                                     |
# +-------------------------------------------------------------------------------------------------+

import pandas as pd
import numpy as np
import os
import csv
import sys

from benchmarks.mimic.util import *

### functions to read and store data in a pandas dataframe: 
#   1) given a path to the csv file, it reads csv to pandas
#   2) select the relevant columns
#   3) convert temporal features to a datetime


def read_patients_table(mimic3_path):
    pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    pats.DOB = pd.to_datetime(pats.DOB)
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats


def read_admissions_table(mimic3_path):
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits


# augmented data collection -> new variables: INSURANCE, LANGUAGE, RELIGION, MARITAL_STATUS
def read_admissions_table_augmented(mimic3_path):
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'DIAGNOSIS']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits


def read_icustays_table(mimic3_path):
    stays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays


def read_icd_diagnoses_table(mimic3_path):
    # load the dictionary for the codes
    codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
    codes = codes[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']]
    # load the diagnoses based on ICD codes
    diagnoses = dataframe_from_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))
    diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)
    return diagnoses


# specific reader function: 
def read_events_table_by_row(mimic3_path, table):
    nb_rows = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
    reader = csv.DictReader(open(os.path.join(mimic3_path, table.upper() + '.csv'), 'r'))
    for i, row in enumerate(reader):
        if 'ICUSTAY_ID' not in row:
            row['ICUSTAY_ID'] = ''
        yield row, i, nb_rows[table.lower()]
# The yield statement suspends function’s execution and sends a value back to the caller, 
# but retains enough state to enable function to resume where it is left off

# count the number of occurances with the same ICD9 code
# return the counts of codes that have appeared at least once
def count_icd_codes(diagnoses, output_path=None):
    codes = diagnoses[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']].drop_duplicates().set_index('ICD9_CODE')
    codes['COUNT'] = diagnoses.groupby('ICD9_CODE')['ICUSTAY_ID'].count()
    codes.COUNT = codes.COUNT.fillna(0).astype(int)
    codes = codes.loc[codes.COUNT > 0]
    if output_path:
        codes.to_csv(output_path, index_label='ICD9_CODE')
    return codes.sort_values('COUNT', ascending=False).reset_index()


# only keep stays that were not part of a admission containing a transfer
# do not need FIRST & LAST_WARDID plus FIRST_CAREUNIT
def remove_icustays_with_transfers(stays):
    stays = stays.loc[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
    return stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]


# inner join of two tables on subjectid
def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])


# inner join on subjectid and admissionid
def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


#-------------------------------- add additional features to stays dataframe ----------------------------------------------------

# DOB: date of birth
def add_age_to_icustays(stays):
    stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
    stays.loc[stays.AGE < 0, 'AGE'] = 90     # DOB of patients over 89 where shifted to >300 years
    return stays


def add_age_to_icustays_revised(stays):
    stays['AGE'] = stays.apply(lambda e: (e['INTIME'] - e['DOB']).days/365, axis=1)
    stays.loc[stays.AGE < 0, 'AGE'] = 90     # DOB of patients over 89 where shifted to >300 years
    return stays


# DOD: date of death
def add_inhospital_mortality_to_icustays(stays):
    # check that person died and death occured during hospitalization
    mortality = stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD))
    # combine with DEATHTIME (seems to be second feature equivalent to DOD)
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME)))
    stays['MORTALITY'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    return stays


# look at INTIME and OUTTIME instead of ADMITTIME / DISCHTIME
def add_inunit_mortality_to_icustays(stays):
    mortality = stays.DOD.notnull() & ((stays.INTIME <= stays.DOD) & (stays.OUTTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME)))
    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays


#----------------------------------- filtering functions -----------------------------------------------------------------

def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    # count the number of icustays per HADM_ID
    to_keep = stays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
    to_keep = to_keep.loc[(to_keep.ICUSTAY_ID >= min_nb_stays) & (to_keep.ICUSTAY_ID <= max_nb_stays)][['HADM_ID']]
    stays = stays.merge(to_keep, how='inner', left_on='HADM_ID', right_on='HADM_ID')
    return stays


def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays.loc[(stays.AGE >= min_age) & (stays.AGE <= max_age)]
    return stays


# only keep diagnoses associated with the retained stays; add ICUSTAY_ID
def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates(), how='inner',
                           left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


#------------------------------------- break-up functions -------------------------------------------------------------------

def break_up_stays_by_subject(stays, output_path, subjects=None, verbose=1):
    # select the subjects to study: either all unique ones or prespecified
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    
    if verbose:
        print('BREAKING UP STAYS BY SUBJECT:')
    
    # for each row / subject: try to create a new directory with personal name
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        
        # write all stays associated with given subject to created directory
        stays.loc[stays.SUBJECT_ID == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'stays.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')


# same procedure: create individual diagnoses csv files per subject
def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None, verbose=1):
    subjects = diagnoses.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    
    if verbose:
        print('BREAKING UP DIAGNOSES BY SUBJECT:')
    
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses.loc[diagnoses.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID','SEQ_NUM'])\
        .to_csv(os.path.join(dn,'diagnoses.csv'), index=False)
    
    if verbose:
        sys.stdout.write('DONE!\n')

# UOM: unit of measurement?
def read_events_table_and_break_up_by_subject(mimic3_path, table, output_path, items_to_keep=None, subjects_to_keep=None, verbose=1):
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
    
    # create sets of items and subjects to keep
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.last_write_no = 0
            self.last_write_nb_rows = 0
            self.last_write_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    ##################################################################################################

    def write_current_observations():
        data_stats.last_write_no += 1
        data_stats.last_write_nb_rows = len(data_stats.curr_obs)
        data_stats.last_write_subject_id = data_stats.curr_subject_id
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))     # dn: directory name
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')                                 # fn: file name

        # if file does not yet exist: open it and define rows given by obs_header
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()

        # class DictWriter: Create object which operates like a regular writer but maps dicts onto output rows
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    ##################################################################################################
    
    for row, row_no, nb_rows in read_events_table_by_row(mimic3_path, table):
        if verbose and (row_no % 100000 == 0):
            if data_stats.last_write_no != '':
                sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...last write '
                                 '({3}) {4} rows for subject {5}'.format(table, row_no, nb_rows,
                                                                         data_stats.last_write_no,
                                                                         data_stats.last_write_nb_rows,
                                                                         data_stats.last_write_subject_id))
            else:
                sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...'.format(table, row_no, nb_rows))

        # check whether given row should be kept based on subjectid and itemid
        if (subjects_to_keep is not None) and (row['SUBJECT_ID'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None) and (row['ITEMID'] not in items_to_keep):
            continue

        # create dictionary
        row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                   'HADM_ID': row['HADM_ID'],
                   'ICUSTAY_ID': '' if 'ICUSTAY_ID' not in row else row['ICUSTAY_ID'],
                   'CHARTTIME': row['CHARTTIME'],
                   'ITEMID': row['ITEMID'],
                   'VALUE': row['VALUE'],
                   'VALUEUOM': row['VALUEUOM']}
        
        # call function defined in this class and save data retrieved
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['SUBJECT_ID']

    # outside of for loop: store last observation
    if data_stats.curr_subject_id != '':
        write_current_observations()

    if verbose:
        sys.stdout.write('\rfinished processing {0}: ROW {1} of {2}...last write '
                         '({3}) {4} rows for subject {5}...DONE!\n'.format(table, row_no, nb_rows,
                                                                           data_stats.last_write_no,
                                                                           data_stats.last_write_nb_rows,
                                                                           data_stats.last_write_subject_id))
