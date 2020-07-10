# +-------------------------------------------------------------------------------------------------+
# | demographics.py: analyse MIMIC root and IHM cohort demographics for reporting in paper          |
# |                                                                                                 |
# | Eliane Röösli (2020), adapted from Harutyunyan et al (2019)                                     |
# +-------------------------------------------------------------------------------------------------+

import pandas as pd
import numpy as np
import os
import sys

from benchmarks.mimic.util import *

#-------------------------------------------- adapted from mimic3csv.py ----------------------------------------------------------------------#


def read_icustays_table(mimic3_path):
    stays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
    stays = stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS']]
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays


# DOB: date of birth
def add_age_to_icustays(stays):
    stays['AGE'] = stays.apply(lambda e: (e['INTIME'] - e['DOB']).days/365, axis=1)
    stays.loc[stays.AGE < 0, 'AGE'] = 90     
    return stays



#----------------------------------------------------------- additional functions for analysis ------------------------------------------------------#

    
def nbr_stays(df, lower, upper, verbose=False):
    group_tot = df[(df['AGE']>=lower) & (df['AGE']<upper)].shape[0]
    return group_tot

def ihm_stays(df, lower, upper, verbose=False):
    ihm_age = ihm(df[(df['AGE']>=lower) & (df['AGE']<upper)])    
    return ihm_age


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
        print("-"*30)


        
############ STATS functions        
        
def demographic_stats(df, report=False):
    total = df.shape[0]
    
    total_F = df[df['GENDER']==1].shape[0]
    total_M = df[df['GENDER']==2].shape[0]
    
    #total_neo = nbr_stays(df,-1,1)
    #total_kids = nbr_stays(df,1,13) 
    #total_teens = nbr_stays(df,13,18)
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
    
    #total_gov = df[df['INSURANCE']==1].shape[0]
    total_medicare = df[df['INSURANCE']==2].shape[0]
    total_medicaid = df[df['INSURANCE']==3].shape[0]
    total_private = df[df['INSURANCE']==4].shape[0]
    #total_self = df[df['INSURANCE']==5].shape[0]
    #total_unknown = df[df['INSURANCE']==0].shape[0]
    total_otherins = df[(df['INSURANCE']==0) | (df['INSURANCE']==1) | (df['INSURANCE']==5)].shape[0]
    
    for name, value in locals().items():
        if "total" in name:
            print('{0:15}  {1:10} ({2:3.1f})'.format(name, value, value/total*100))
            
    if report:
        print('FOR REPORTING:')
        for name, value in locals().items():
            if "total" in name:
                print('{0} ({1:3.1f})'.format(value, value/total*100))
            
def ihm(df):
    return np.round(df['IHM'].sum()/df.shape[0],4)
            
            
def IHM_stats(df, report=False):
    ihm_total = ihm(df)
    ihm_total_ratio = np.round(df['IHM'].sum()/(df.shape[0]-df['IHM'].sum()),4)
    
    ihm_F = ihm(df[df['GENDER']==1])
    ihm_M = ihm(df[df['GENDER']==2])
    
    ihm_neo = ihm_stays(df,-1,1)
    ihm_kids = ihm_stays(df,1,13) 
    ihm_teens = ihm_stays(df,13,18)
    ihm_adults1 = ihm_stays(df,18,30)
    ihm_adults2 = ihm_stays(df,30,50)
    ihm_adults3 = ihm_stays(df,50,70)
    ihm_seniors1 = ihm_stays(df,70,90)
    ihm_seniors2 = ihm_stays(df,90, 400)

    ihm_asian = ihm(df[df['ETHNICITY']==1])
    ihm_black = ihm(df[df['ETHNICITY']==2])
    ihm_hispanic = ihm(df[df['ETHNICITY']==3])
    ihm_white = ihm(df[df['ETHNICITY']==4])
    ihm_other = ihm(df[df['ETHNICITY']==0])
    
    ihm_gov = ihm(df[df['INSURANCE']==1])
    ihm_medicare = ihm(df[df['INSURANCE']==2])
    ihm_medicaid = ihm(df[df['INSURANCE']==3])
    ihm_private = ihm(df[df['INSURANCE']==4])
    ihm_self = ihm(df[df['INSURANCE']==5])
    ihm_unknown = ihm(df[df['INSURANCE']==0])
    ihm_otherins = ihm(df[(df['INSURANCE']==0) | (df['INSURANCE']==1) | (df['INSURANCE']==5)])
    
    for name, value in locals().items():
        if "ihm" in name:
            print('{0:15}  {1:4.2f}'.format(name, value*100))
            
    if report:
        print('FOR REPORTING:')
        for name, value in locals().items():
            if "ihm" in name:
                print('{0:4.1f}\%'.format(value*100))

                
                
                
############ MIMIC root cohort                
                
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



############ MIMIC IHM cohort


def mimic_cohort_demographics(path, option):
    if option == 'train':
        return partial_cohort(path, 'train_listfile.csv', option)
    elif option == 'test':
        return partial_cohort(path, 'test_listfile.csv', option)
    elif option == 'total':
        return total_cohort(os.path.join(path, 'demographics'))
    else:
        print('wrong option chosen')
        

def total_cohort(task_path_dem):
    df_patients = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    df_stays = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    episodes = os.listdir(task_path_dem)
    nb_episodes = len(episodes)        
    seen_patients = []
    for (episode_index, episode) in enumerate(episodes):
        sys.stdout.write('\rEPISODE {0} of {1}...'.format(episode_index+1, nb_episodes))
        if episode.endswith(".csv"):
            data = pd.read_csv(os.path.join(task_path_dem,episode))
            patient_id = episode.split('_')[0]
            if patient_id not in seen_patients:
                seen_patients.append(patient_id)
                df_patients = df_patients.append({'ETHNICITY': data['Ethnicity'].iloc[0], 'GENDER': data['Gender'].iloc[0], 
                                                'AGE': data['Age'].iloc[0], 'INSURANCE': data['Insurance'].iloc[0],
                                                    'IHM': data['Mortality'].iloc[0]}, ignore_index=True)

            df_stays = df_stays.append({'ETHNICITY': data['Ethnicity'].iloc[0], 'GENDER': data['Gender'].iloc[0], 
                                                'AGE': data['Age'].iloc[0], 'INSURANCE': data['Insurance'].iloc[0],
                                                     'IHM': data['Mortality'].iloc[0]}, ignore_index=True)
                
    return df_patients, df_stays


def partial_cohort(task_path, file, option):
    listfile = pd.read_csv(os.path.join(task_path, file))
        
    df_patients = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    df_stays = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    seen_patients = []
    nb_episodes = listfile.shape[0]
    
    print('chosen option:', option)
    for (episode_index, episode) in enumerate(listfile['stay']):
            sys.stdout.write('\rEPISODE {0} of {1}...'.format(episode_index+1, nb_episodes))
            if episode.endswith(".csv"):
                data = pd.read_csv(os.path.join(task_path, 'demographics', episode.replace('_timeseries', '')))
                patient_id = episode.split('_')[0]
                stay_id = episode.split('_')[1].replace('episode', '')
                
            
                if patient_id not in seen_patients:
                    seen_patients.append(patient_id)
                    df_patients = df_patients.append({'PAT_DEID': patient_id, 'ETHNICITY': data['Ethnicity'].iloc[0], 'GENDER': data['Gender'].iloc[0], 
                                                    'AGE': data['Age'].iloc[0], 'INSURANCE': data['Insurance'].iloc[0],
                                                        'IHM': data['Mortality'].iloc[0]}, ignore_index=True)

                df_stays = df_stays.append({'PAT_DEID': patient_id, 'STAY_ID': stay_id, 'CSV': episode, 'ETHNICITY': data['Ethnicity'].iloc[0], 'GENDER': data['Gender'].iloc[0], 
                                                    'AGE': data['Age'].iloc[0], 'INSURANCE': data['Insurance'].iloc[0],
                                                         'IHM': data['Mortality'].iloc[0]}, ignore_index=True)
    return df_patients, df_stays


############ STARR IHM cohort

def starr_cohort_demographics(path, option):
    stays = pd.read_csv(os.path.join(path, 'stays.tsv'), sep='\t')
    if option == 'train':
        path_file = 'ihm/train_listfile.csv'
    elif option == 'test':
        path_file = 'ihm/test_listfile.csv'
    elif option == 'total':
        path_file = 'ihm/listfile.csv'  
    else:
        print('wrong option chosen')
        
    listfile = pd.read_csv(os.path.join(path, path_file))
        
    df_patients = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    df_stays = pd.DataFrame(columns = ['ETHNICITY', 'GENDER', 'AGE', 'INSURANCE'])
    seen_patients = []
    nb_episodes = listfile.shape[0]
    
    print('chosen option:', option)
    for (episode_index, episode) in enumerate(listfile['stay']):
            sys.stdout.write('\rEPISODE {0} of {1}...'.format(episode_index+1, nb_episodes))
            patient_id = episode.split('_')[0]
            stay_id = episode.split('_')[1].replace('episode', '')

            data = stays[(stays['pat_deid'] == int(patient_id)) & (stays['stay_id'] == int(stay_id))]
            if data.shape[0] == 1:
                if patient_id not in seen_patients:
                    seen_patients.append(patient_id)
                    df_patients = df_patients.append({'ETHNICITY': data['race'].iloc[0], 'GENDER': data['gender'].iloc[0], 
                                                    'AGE': data['age'].iloc[0], 'INSURANCE': data['insurance'].iloc[0],
                                                     'IHM': data['ihm'].iloc[0]}, ignore_index=True)

                df_stays = df_stays.append({'ETHNICITY': data['race'].iloc[0], 'GENDER': data['gender'].iloc[0], 
                                                'AGE': data['age'].iloc[0], 'INSURANCE': data['insurance'].iloc[0],
                                                'IHM': data['ihm'].iloc[0]}, ignore_index=True)
                
    return df_patients, df_stays