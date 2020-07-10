# +-------------------------------------------------------------------------------------------------+
# | preprocessing.py: functions to clean up stays, labs and vitals data                             |
# |                                                                                                 |
# | Eliane Röösli (2020)                                                                            |
# +-------------------------------------------------------------------------------------------------+

import numpy as np
import pandas as pd


#----------------------------------------- VARIABLES -----------------------------------------

def variable_list(var_map):
    v_list = list(var_map.values())
    v_list.insert(0, 'Hours')
    return v_list

var_map = {
    'CRR': 'Capillary refill rate', 
    'DBP': 'Diastolic blood pressure', 
    'ETH': 'Ethnicity', 
    'FIO2': 'Fraction inspired oxygen', 
    'GEN': 'Gender', 
    'GCSE': 'Glascow coma scale eye opening',
    'GCSM': 'Glascow coma scale motor response',
    'GCST': 'Glascow coma scale total', 
    'GCSV': 'Glascow coma scale verbal response', 
    'GLU': 'Glucose', 
    'HR': 'Heart rate', 
    'HT': 'Height', 
    'INS': 'Insurance', 
    'MBP': 'Mean blood pressure', 
    'O2S': 'Oxygen saturation',
    'RR': 'Respiratory rate',
    'SBP': 'Systolic blood pressure',
    'TP': 'Temperature',
    'WT': 'Weight',
    'PH': 'pH'  
}


#---------------------------------------- CLEAN & UPDATE STAY VALUES: HEIGHT & WEIGHT -------------------------------------

def clean_stays(stays, vitals, verbose=False):
    
    if verbose:
        print('-'*80)
        print('CLEAN STAYS')
    
    # clean charlson comorbidity score
    stays = clean_charlson(stays, verbose)
    
    # clean height and weight data from encounter table
    stays = clean_height(stays, verbose)
    stays = clean_weight(stays, verbose)
    
    # complete height and weight data from vitals
    stays = add_height(stays, vitals, verbose)
    stays = add_weight(stays, vitals, verbose)
    
    return stays


def clean_charlson(stays, verbose=False):
    # prepare handling of null values: correspond in fact to 0
    stays['comorb1'].replace("None", 0, inplace=True)
    stays['comorb2'].replace("None", 0, inplace=True)
    
    # convert strings to floats
    stays['comorb1'] = stays['comorb1'].apply(lambda g: int(g))
    stays['comorb2'] = stays['comorb2'].apply(lambda g: int(g))
    
    return stays


def clean_height(stays, verbose=False):
    # prepare handling of null values
    stays['height'].replace("None", "NaN\' NaN", inplace=True)
    
    # need to split feet and inches before converting it to centimeters
    feet = stays['height'].apply(lambda x: float(str(x).split('\'', 1)[0]))
    inches = stays['height'].apply(lambda x: float(str(x).split('\'', 1)[1].replace('"', '').strip()))
    cm = np.round(30.48*feet + 2.54*inches,2)
    stays['height'] = cm
    
    if verbose:
        print('number of stays with height data: {}'.format(stays['height'].count()))

    return stays


def clean_weight(stays, verbose=False):
    # prepare handling of null values
    stays['weight'].replace("None", np.nan, inplace=True)
    
    # convert oz to kg
    stays['weight'] = stays['weight'].apply(lambda x: np.round(float(x)*28.349523125/1000,2))
    
    if verbose:
        print('number of stays with weight data: {}'.format(stays['weight'].count()))

    return stays


def add_height(stays, vitals, verbose=False):
    table_HT = new_data(stays, vitals, 'Height', verbose)
    
    for idx, h in table_HT.iterrows():
        stays.loc[idx,'height'] = np.round(float(h['value'])*25.4/10,2) 

    return stays


def add_weight(stays, vitals, verbose=False):
    table_WT = new_data(stays, vitals, 'Weight', verbose)
    
    for idx, w in table_WT.iterrows():
        stays.loc[idx, 'weight'] = np.round(float(w['value'])*28.349523125/1000,2) 
   
    return stays


def new_data(stays, vitals, variable, verbose=False):
    table_var = vitals[vitals['event_id']==variable].copy()
    table_var.sort_values('hours', ascending=True, inplace=True)
    table_var.drop_duplicates(subset=['pat_deid', 'stay_id'], keep='first', inplace=True)
    table_merged = pd.merge(stays, table_var, how='outer', on=['pat_deid', 'stay_id'])
    table_new = table_merged[table_merged.apply(lambda x: ((x['value'] != 'None') and 
                                             (str(x['value']).find('nan') == -1) and 
                                             (str(x[variable.lower()]).find('nan') != -1)), axis=1)]
    if verbose:
        print('number additional measures for {}: {}'.format(variable, table_new.shape[0]))
        
    return table_new




#------------------------------------------- CLEAN LAB VALUES -------------------------------


def clean_labs(labs, verbose=False):
    
    if verbose:
        print('-'*80)
        print('CLEAN LABS')
    
    # remove None and np.nan values
    if verbose:
        print('removed {} lab measures with missing value'.format(labs[labs['value'].apply(lambda v: (v == 'None' or v == np.nan))].shape[0]))
    labs = labs[labs['value'].apply(lambda g: (g != 'None' and g != np.nan))].copy()
    
    # clean variables that need cleaning (not necessary: o2sat, heartrate)
    labs = clean_glucose(labs)
    labs = clean_pH(labs)
    labs = clean_fio2(labs)
    
    # convert all values to floats (from strings)
    labs['value'] = labs['value'].apply(lambda g: float(g))
    
    return labs


# cleaning functions for lab values

def clean_glucose(labs):
    mapping = {'<20': '19', '>500': '501', '>600': '601'}
    labs['value'] = labs.apply(lambda g: mapping[g['value']] if g['value'] in mapping.keys() and g['event_id']=='Glucose' else g['value'], axis=1)
    return labs

def clean_pH(labs):
    labs['value'] = labs.apply(lambda g: '6.50' if g['value'] == '<6.50' and g['event_id'] == 'pH' else g['value'], axis=1)
    return labs

def clean_fio2(labs):
    labs['value'] = labs.apply(lambda g: float(g['value'])/100 if float(g['value']) > 1 and g['event_id'] == 'Fraction inspired oxygen' else g['value'], axis=1)
    labs['value'] = labs.apply(lambda g: 0.21 if float(g['value']) < 0.21 and g['event_id'] == 'Fraction inspired oxygen' else g['value'], axis=1)
    return labs



#---------------------------------------- CLEAN VITALS ----------------------------------------------------

def clean_vitals(vitals, verbose=False):
    
    if verbose:
        print('-'*80)
        print('CLEAN VITALS')
    
    # exclude None / np.nan values and height/weight data (already incorporated into stays data)
    vitals = exclude_vitals(vitals, verbose)
    
    # clean temperature, blood pressure and GCS measurements
    vitals = clean_TP(vitals)
    vitals = clean_BP(vitals)
    vitals = clean_GCS(vitals)
    
    # convert all values to floats (from strings)
    vitals['value'] = vitals.apply(lambda g: float(g['value']) if g['event_id'].find('Glascow') == -1 else g['value'], axis=1)
    
    return vitals


def exclude_vitals(vitals, verbose=False):
    
    if verbose:
        print('removed {} vitals measures with missing value'.format(vitals[vitals['value'].apply(lambda v: (v == 'None' or v == np.nan))].shape[0]))
        print('removed remaining {} heights and {} weights measures from vitals'.format(vitals[vitals['event_id'] == 'Height'].shape[0], 
                                                                                      vitals[vitals['event_id'] == 'Weight'].shape[0]))
        
    # exclude measurement with None or np.nan value
    vitals = vitals[vitals['value'].apply(lambda v: (v != 'None' and v != np.nan))]
    
    # exclude height and weight measurements
    vitals = vitals[vitals['event_id'].apply(lambda v: (v != 'Height' and v != 'Weight'))]

    return vitals
    
    
### Temperature

def clean_TP(vitals):
    vitals['value'] = vitals.apply(lambda x: np.round(helper_TP(x),2) if x['event_id'] == 'Temperature' else x['value'], axis=1)
    return vitals

def helper_TP(t):
    # if format is 'xx.x ?C (xx.x ?F)'
    if t['value'].find('?C') != -1:
        return float(t['value'].split('?C',1)[0].strip())
    # if value is None
    elif t['value'] == 'None':
        return np.nan
    # if value is already in celsius
    elif 25 < float(t['value']) < 45:
        return float(t['value'])
    # if value is in fahrenheit
    elif 80 < float(t['value']) < 110:
        return (float(t['value'])-32)*5/9
    else:
        return np.nan
    
    
### Blood pressure

def clean_BP(vitals):
    # split vitals into BP measures and other vitals
    non_BP = vitals[vitals['name'] != 'BP']
    BP = vitals[vitals['name'] == 'BP'].copy()
    
    # prepare handling of null values
    BP['value'].replace("None", "NaN/NaN", inplace=True)

    # extract sbp and dbp values, calculate mbp values
    sbp_values = BP['value'].apply(lambda x: float(x.split('/', 1)[0]) if x != np.nan and x.find('/') != -1 else np.nan)
    dbp_values = BP['value'].apply(lambda x: float(x.split('/', 1)[1]) if x != np.nan and x.find('/') != -1 else np.nan)
    mbp_values = (sbp_values + 2*dbp_values)/3
    mbp_values = [np.round(val,2) for val in mbp_values]

    # update values and store it in individual tables
    SBP = convert_BPtable(BP, sbp_values, 'Systolic blood pressure')
    DBP = convert_BPtable(BP, dbp_values, 'Diastolic blood pressure')
    MBP = convert_BPtable(BP, mbp_values, 'Mean blood pressure')
    
    return pd.concat([non_BP, SBP, DBP, MBP], axis=0).reset_index(drop=True)


def convert_BPtable(table, values, event_id):
    new_table = table.copy()
    new_table['value'] = values
    new_table['event_id'] = event_id
    return new_table



### clean Glasgow coma scale

def clean_GCS(vitals):
    vitals['value'] = vitals.apply(lambda x: helper_GCS(x) if (x['event_id'].find("Glascow") != -1) else x['value'], axis=1)
    return vitals


def helper_GCS(gcs):
    val = gcs['value']
    if gcs['event_id'].find("verbal") != -1:
        return GCS_verbal[str(val)]
    elif gcs['event_id'].find("eye") != -1:
        return GCS_eye[str(val)]
    elif gcs['event_id'].find("motor") != -1:
        return GCS_motor[str(val)]
    elif gcs['event_id'].find("total") != -1:
        return str(int(val))
    else:
        return np.nan
    
    
GCS_verbal = {
    '1': '1 No Response',
    '2': '2 Incomp sounds',
    '3': '3 Inapprop words',
    '4': '4 Confused',
    '5': '5 Oriented'
}

GCS_eye = {
    '1': '1 No Response',
    '2': '2 To pain',
    '3': '3 To speech',
    '4': '4 Spontaneously'
}

GCS_motor = {
    '1': '1 No Response',
    '2': '2 Abnorm extensn',
    '3': '3 Abnorm flexion',
    '4': '4 Flex-withdraws',
    '5': '5 Localizes Pain',
    '6': '6 Obeys Commands'
}