from __future__ import absolute_import
from __future__ import print_function

import argparse
import yaml
import os

from mimic3benchmark.mimic3csv import *
from mimic3benchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from mimic3benchmark.util import *

### get arguments from user

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                    default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'])
parser.add_argument('--phenotype_definitions', '-p', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccs_2015_definitions.yaml'),
                    help='YAML file with phenotype definitions.')
parser.add_argument('--itemids_file', '-i', type=str, help='CSV containing list of ITEMIDs to keep.')
parser.add_argument('--augmented', action='store_true', help='AUGMENTED: using additional demographic variables')
parser.add_argument('--verbose', '-v', type=int, help='Level of verbosity in output.', default=1)
parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1000 subjects, 1000000 events.')
args, _ = parser.parse_known_args()

try:
    os.makedirs(args.output_path)
except:
    pass

### 1. read three main tables

patients = read_patients_table(args.mimic3_path)
stays = read_icustays_table(args.mimic3_path)

if args.augmented:
    print('AUGMENTED DATASET: USING MORE DEMOGRAPHIC VARIABLES')
    admits = read_admissions_table_augmented(args.mimic3_path)
else: 
    admits = read_admissions_table(args.mimic3_path)

if args.verbose:
    print('1. {0:40} (1) ICUSTAY_ID: {1}  (2) HADM_ID: {2}  (3) SUBJECT_ID: {3}'
          .format('NUMBER OF UNIQUE SAMPLES:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

### 2. remove icustays that had transfers

stays = remove_icustays_with_transfers(stays)
if args.verbose:
    print('2. {0:40} (1) ICUSTAY_ID: {1}  (2) HADM_ID: {2}  (3) SUBJECT_ID: {3}'
          .format('AFTER REMOVING ICU TRANSFERS:',stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

### 3. remove multiple stay per admission

# functions are from mimic3csv: are inner joins
stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)
stays = filter_admissions_on_nb_icustays(stays)
if args.verbose:
    print('3. {0:40} (1) ICUSTAY_ID: {1}  (2) HADM_ID: {2}  (3) SUBJECT_ID: {3}'
          .format('AFTER REMOVING MULTIPLE STAYS PER ADMIT:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

### 4. add additional features and remove neonates

stays = add_age_to_icustays(stays)
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
stays = filter_icustays_on_age(stays)
if args.verbose:
    print('4. {0:40} (1) ICUSTAY_ID: {1}  (2) HADM_ID: {2}  (3) SUBJECT_ID: {3}'
          .format('AFTER REMOVING PATIENTS AGE < 18:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

### 5. write cleaned stays to csv

if args.verbose:
    print('INCLUDED VARIABLES IN STAYS TABLE:\n', list(stays.columns))
    print('5. WRITE STAYS TO CSV')
stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)


### 6. get diagnosis data

if args.verbose:
    print('6. GET DIAGNOSIS DATA AND CREATE PHENOTYPES FROM IT')
diagnoses = read_icd_diagnoses_table(args.mimic3_path)
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)
count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))
phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.load(open(args.phenotype_definitions, 'r')))
make_phenotype_label_matrix(phenotypes, stays).to_csv(os.path.join(args.output_path, 'phenotype_labels.csv'),index=False, quoting=csv.QUOTE_NONNUMERIC)


### 7. if in test mode: randomly select data from 1000 patients

if args.test:
    pat_idx = np.random.choice(patients.shape[0], size=1000)
    patients = patients.iloc[pat_idx]
    stays = stays.merge(patients[['SUBJECT_ID']], left_on='SUBJECT_ID', right_on='SUBJECT_ID')
    args.event_tables = [args.event_tables[0]]
    if args.verbose:
        print('7. IN TEST MODE: USING ONLY {0} STAYS AND ONLY {1} TABLE'.format(stays.shape[0], args.event_tables[0]))


### 8. final data prep (TAKES THE MOST TIME)

if args.verbose:
    print('8. PREPARE SUBDIRECTORIES AND INDIVIDUAL DATA FILES')

# get all unique subjects
subjects = stays.SUBJECT_ID.unique()
              
# assign stay to subjects
break_up_stays_by_subject(stays, args.output_path, subjects=subjects, verbose=args.verbose)
              
# assign diagnoses to subjects
break_up_diagnoses_by_subject(phenotypes, args.output_path, subjects=subjects, verbose=args.verbose)
              
# assign events to subjeCts
if args.itemids_file:
    items_to_keep = set([int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['ITEMID'].unique()])
else: 
    items_to_keep = None

# there are three different events tables: CHARTEVENTS, LABEVENTS, OUTPUTEVENTS
for table in args.event_tables:
    read_events_table_and_break_up_by_subject(args.mimic3_path, table, args.output_path, items_to_keep=items_to_keep,
                                              subjects_to_keep=subjects, verbose=args.verbose)
