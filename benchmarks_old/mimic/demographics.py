import os
import sys
import pandas as pd
import numpy as np
from mimic3benchmark.util import *

#-------------------------------------------- adapted from mimic3csv.py ----------------------------------------------------------------------#

def read_patients_table(mimic3_path):
    pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    pats.DOB = pd.to_datetime(pats.DOB)
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats


def read_admissions_table(mimic3_path):
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'DIAGNOSIS']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits


def read_icustays_table(mimic3_path):
    stays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
    stays = stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS']]
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays

def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])


# inner join on subjectid and admissionid
def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


### add additional features to stays dataframe

# DOB: date of birth
def add_age_to_icustays(stays):
    #stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
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


def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    # count the number of icustays per HADM_ID
    to_keep = stays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
    to_keep = to_keep.loc[(to_keep.ICUSTAY_ID >= min_nb_stays) & (to_keep.ICUSTAY_ID <= max_nb_stays)][['HADM_ID']]
    stays = stays.merge(to_keep, how='inner', left_on='HADM_ID', right_on='HADM_ID')
    return stays



#----------------------------------------------------------- additional functions for analysis ------------------------------------------------------#

def nbr_stays(lower, upper, group_name):
    group_tot = stays[(stays['AGE']>lower) & (stays['AGE']<=upper)].shape[0]
    print('total number of stays by {0}: {1}'.format(group_name, group_tot))
    print('percentage of stays by {0} in MIMIC: {1:.{2}f}%'.format(group_name, group_tot/stays_tot*100,2))
    

def grouping(x, groups, names):
    for j, group in enumerate(groups):
        group_name = names[j]
        if x in group:
            return group_name
        

def stat_by_group(df, var, group_bounds, group_list):
    stats_list = []
    for i, bounds in enumerate(group_bounds):
        stats_list.append(stats(df,var,bounds))
    x = np.arange(0,len(stats_list))
    f, ax = plt.subplots(figsize=(15,5))
    plt.bar(x=x,height=stats_list, width=0.6, tick_label=group_list)
    plt.ylabel('number of observations')
    plt.xlabel('population groups by age')
    
    return stats_list


def stats(df, var, bounds):
    return df[(np.floor(df[var])>bounds[0]) & (np.floor(df[var]<=bounds[1]))].shape[0]


#-------------------------------------------------------------- transform demographic variables -------------------------------------------

g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}
def transform_gender(gender_series):
    global g_map    # to access the variable defined outside of the function
    return gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER'])

# ethnicity map: there are many more subtypes but they all contain one of these keywords at first position
e_map = {'ASIAN': 1,
         'BLACK': 2,
         'CARIBBEAN ISLAND': 2,
         'HISPANIC': 3,
         'SOUTH AMERICAN': 3,
         'WHITE': 4,
         'MIDDLE EASTERN': 4,
         'PORTUGUESE': 4,
         'AMERICAN INDIAN': 0,
         'NATIVE HAWAIIAN': 0,
         'UNABLE TO OBTAIN': 0,
         'PATIENT DECLINED TO ANSWER': 0,
         'UNKNOWN': 0,
         'OTHER': 0,
         '': 0}

def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        # replace(old, new); split(separator) returns list of strings: only take first one
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])

i_map = {'Government':1,
         'Medicare': 2,
         'Medicaid': 3,
         'Private': 4,
         'Self Pay': 5,
         'Other': 0,
         '': 0}

def transform_insurance(insurance_series):
    global i_map
    return insurance_series.fillna('').apply(lambda s: i_map[s] if s in i_map else i_map['Other'])


#---------------------------------------------------- demographics ------------------------------------------------------------

def detailed_ethnicity(stays):
    # some redefinition
    stays.loc[stays["ETHNICITY"] == "UNABLE TO OBTAIN", "ETHNICITY"] = "UNKNOWN/NOT SPECIFIED"
    stays.loc[stays["ETHNICITY"] == "PATIENT DECLINED TO ANSWER", "ETHNICITY"] = "UNKNOWN/NOT SPECIFIED"
    stays.loc[stays["ETHNICITY"] == "AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE", "ETHNICITY"] = "AMERICAN INDIAN/ALASKA NATIVE"
    groups = np.arange(0,5)
    labels = ["UNKNOWN/OTHER", "ASIAN", "BLACK", "HISPANIC", "WHITE"]
    total = stays.shape[0]
    print(total)
    for g in groups:
        grouptotal = stays[stays["ETHNICITY_"] == g].shape[0]
        print("{} {} ({:4.2f}) ".format(labels[g], grouptotal, grouptotal/total*100,2))
        
        cols = stays[stays["ETHNICITY_"] == g]["ETHNICITY"].unique()
        for c in cols:
            small_total = stays[stays['ETHNICITY'] == c].shape[0]
            print("{} {} ({:4.2f})".format(c, small_total, small_total/grouptotal*100,2))
        print("-----------------------------")




def nbr_stays(df, lower, upper):
    return df[(df['AGE']>=lower) & (df['AGE']<upper)].shape[0]


def demographic_stats(df, report=False):
    total = df.shape[0]
    
    total_F = df[df['GENDER']==1].shape[0]
    total_M = df[df['GENDER']==2].shape[0]
    
    total_neo = nbr_stays(df,-1,1)
    total_kids = nbr_stays(df,1,13) 
    total_teens = nbr_stays(df,13,18)
    total_adults1 = nbr_stays(df,18,30)
    total_adults2 = nbr_stays(df,30,50)
    total_adults3 = nbr_stays(df,50,70)
    total_seniors1 = nbr_stays(df,70,90)
    total_seniors2 = nbr_stays(df,90, 400)

    total_asian = df[df['ETHNICITY']==1].shape[0] 
    total_black = df[df['ETHNICITY']==2].shape[0]
    total_hispanic = df[df['ETHNICITY']==3].shape[0]
    total_white = df[df['ETHNICITY']==4].shape[0]
    total_other = df[df['ETHNICITY']==0].shape[0]
    
    total_gov = df[df['INSURANCE']==1].shape[0]
    total_medicare = df[df['INSURANCE']==2].shape[0]
    total_medicaid = df[df['INSURANCE']==3].shape[0]
    total_private = df[df['INSURANCE']==4].shape[0]
    total_self = df[df['INSURANCE']==5].shape[0]
    
    for name, value in locals().items():
        if "total" in name:
            print('{0:15}  {1:10} ({2:4.2f})'.format(name, value, value/total*100))
            
    if report:
        print('FOR REPORTING:')
        for name, value in locals().items():
            if "total" in name:
                print('{0} ({1:4.2f})'.format(value, value/total*100))
            
    
            
def rootcohort_demographics(root_path):
    
    patients = list(filter(str.isdigit, os.listdir(root_path)))
    nb_patients = len(patients)
    df_patients = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    df_stays = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    print(nb_patients)
    for (patient_index, patient) in enumerate(patients):
        sys.stdout.write('\rSUBJECT {0} of {1}...'.format(patient_index+1, nb_patients))
        
        # get patient timeseries event data from folder
        patient_folder = os.path.join(root_path, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        if patient_ts_files == []:
            print('EMPTY:', patient_folder)
        new_patient = True
        # for each episode during an admission
        for (episode_index, ts_filename) in enumerate(patient_ts_files):
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                # get diagnosis data from the non timeseries csv file
                lb_filename = ts_filename.replace("_timeseries", "")
                data = pd.read_csv(os.path.join(patient_folder, lb_filename))
                
                # index is to make sure that each row is unique (otherwise not added by append function)
                stay_index = (patient_index+1)*100+(episode_index+1)
                df_stays = df_stays.append({'INDEX': stay_index, 'ETHNICITY': data['Ethnicity'].iloc[0], 'GENDER': data['Gender'].iloc[0], 
                                        'AGE': data['Age'].iloc[0], 'INSURANCE': data['Insurance'].iloc[0]}, ignore_index=True)
                if new_patient:
                    df_patients = df_patients.append({'INDEX':patient_index+1, 'ETHNICITY': data['Ethnicity'].iloc[0], 'GENDER': data['Gender'].iloc[0], 
                                        'AGE': data['Age'].iloc[0], 'INSURANCE': data['Insurance'].iloc[0]}, ignore_index=True)
                    new_patient = False
                
    return df_patients, df_stays


def rootcohort_sep_demographics(root_path):
    partitions = ['test', 'train']
    df_patients = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    df_stays = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    for p in partitions:
        patients = list(filter(str.isdigit, os.listdir(os.path.join(root_path,p))))
        nb_patients = len(patients)
        print('\nnumber subjects in', p, 'partition:', nb_patients)
        for (patient_index, patient) in enumerate(patients):
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(patient_index+1, nb_patients))
            # get patient timeseries event data from folder
            patient_folder = os.path.join(root_path,p, patient)
            patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

            new_patient = True
            # for each episode during an admission
            for (episode_index, ts_filename) in enumerate(patient_ts_files):
                with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                    # get diagnosis data from the non timeseries csv file
                    lb_filename = ts_filename.replace("_timeseries", "")
                    data = pd.read_csv(os.path.join(patient_folder, lb_filename))

                    df_stays = df_stays.append({'INDEX':patient+"_"+str(episode_index), 'ETHNICITY': data['Ethnicity'].iloc[0], 'GENDER': data['Gender'].iloc[0], 
                                            'AGE': data['Age'].iloc[0], 'INSURANCE': data['Insurance'].iloc[0]}, ignore_index=True)
                    if new_patient:
                        df_patients = df_patients.append({'INDEX':patient+"_"+str(episode_index), 'ETHNICITY': data['Ethnicity'].iloc[0], 
                                                          'GENDER': data['Gender'].iloc[0], 'AGE': data['Age'].iloc[0], 
                                                          'INSURANCE': data['Insurance'].iloc[0]}, ignore_index=True)
                        new_patient = False
                
    return df_patients, df_stays



def taskcohort_sep_demographics(task_path_dem):
    partitions = ['test', 'train']
    df_patients = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    df_stays = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    for p in partitions:
        print('\nin partition:', p)
        episodes = os.listdir(os.path.join(task_path_dem,p))
        nb_episodes = len(episodes)        
        seen_patients = []
        print(nb_episodes)
        for (episode_index, episode) in enumerate(episodes):
            sys.stdout.write('\rEPISODE {0} of {1}...'.format(episode_index+1, nb_episodes))
            data = pd.read_csv(os.path.join(task_path_dem,p,episode))
            patient_id = episode.split('_')[0]
            if patient_id not in seen_patients:
                seen_patients.append(patient_id)
                df_patients = df_patients.append({'ETHNICITY': data['Ethnicity'].iloc[0], 'GENDER': data['Gender'].iloc[0], 
                                                'AGE': data['Age'].iloc[0], 'INSURANCE': data['Insurance'].iloc[0]}, ignore_index=True)

            df_stays = df_stays.append({'ETHNICITY': data['Ethnicity'].iloc[0], 'GENDER': data['Gender'].iloc[0], 
                                                'AGE': data['Age'].iloc[0], 'INSURANCE': data['Insurance'].iloc[0]}, ignore_index=True)

    return df_patients, df_stays



def taskcohort_demographics(task_path_dem):
    df_patients = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    df_stays = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    episodes = os.listdir(task_path_dem)
    nb_episodes = len(episodes)        
    seen_patients = []
    print(nb_episodes)
    for (episode_index, episode) in enumerate(episodes):
        sys.stdout.write('\rEPISODE {0} of {1}...'.format(episode_index+1, nb_episodes))
        data = pd.read_csv(os.path.join(task_path_dem,episode))
        patient_id = episode.split('_')[0]
        if patient_id not in seen_patients:
            seen_patients.append(patient_id)
            df_patients = df_patients.append({'ETHNICITY': data['Ethnicity'].iloc[0], 'GENDER': data['Gender'].iloc[0], 
                                            'AGE': data['Age'].iloc[0], 'INSURANCE': data['Insurance'].iloc[0]}, ignore_index=True)
        
        df_stays = df_stays.append({'ETHNICITY': data['Ethnicity'].iloc[0], 'GENDER': data['Gender'].iloc[0], 
                                            'AGE': data['Age'].iloc[0], 'INSURANCE': data['Insurance'].iloc[0]}, ignore_index=True)
                
    return df_patients, df_stays