# +-------------------------------------------------------------------------------------------------+
# | create_ihm.py: creates IHM cohort from root cohort                                              |
# |                                                                                                 |
# | Eliane Röösli (2020), adapted from Harutyunyan et al (2019)                                     |
# +-------------------------------------------------------------------------------------------------+

import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
random.seed(49297)


def process_partition(args, partition, eps=1e-6, n_hours=48):
    
    # prepare output directory
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    # prepare demographics directory
    demographics_dir = os.path.join(args.output_path, 'demographics')
    if not os.path.exists(demographics_dir):
        os.mkdir(demographics_dir)

    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    
    print('PARTITION FOR:', partition)
    nb_patients = len(patients)
    patient_included = [0]*nb_patients
    gender_map = {1: 0, 2: 0, 3: 0, 0: 0}
    age_patients = {age: 0 for age in range(18,91)}
    race_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # start with root cohort
    for (patient_index, patient) in enumerate(patients):
        
        sys.stdout.write('\rSUBJECT {0} of {1}...'.format(patient_index+1, nb_patients))
        
        # get patient timeseries event data from folder
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        if patient_ts_files == []:
            print('EMPTY PATIENT')
            
        new_patient = True
        # for each episode during an admission
        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                # get diagnosis data from the non timeseries csv file
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    print("(empty label file)", patient, label_df)
                    continue

                # (1) exclude episodes with missing LOS
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # convert to hours
                if pd.isnull(los):
                    #print("(length of stay is missing)", patient, ts_filename)
                    continue
                
                # (2) exclude episodes shorter than 48h
                if los < n_hours - eps:
                    continue

                # get events during first 48h for a given episode
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]
                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]

                # (3) exclude episodes with no measurements in ICU (during first 48h)
                if len(ts_lines) == 0:
                    #print("(no events in ICU) ", patient, ts_filename)
                    continue

                # collect demographics data if patient is new
                if new_patient:
                    patient_included[patient_index] = 1
                    # gender
                    gender = label_df.iloc[0]['Gender']
                    gender_map[gender] += 1
                    # age
                    age = np.minimum(np.floor(label_df.iloc[0]['Age']), 90)
                    age_patients[age] += 1
                    # race
                    race = label_df.iloc[0]['Ethnicity']
                    race_map[race] += 1
                    # stop recording for additional stays
                    new_patient = False
                
                # store demographics of the stay
                demographics_filename = patient + "_" + lb_filename
                label_df.to_csv(os.path.join(args.output_path, 'demographics', demographics_filename))
                
                # create events storage file
                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                # create target label (patient died before hospital discharge) and (name of feature csv file, label) pairs
                mortality = int(label_df.iloc[0]["Mortality"])
                xy_pairs.append((output_ts_filename, mortality))

    print("\nTOTAL NUMBER ICU STAYS:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)

    # save datasets to output directory
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,y_true\n')
        for (x, y) in xy_pairs:
            listfile.write('{},{:d}\n'.format(x, y))
                
    print("TOTAL NUMBER PATIENTS:", sum(patient_included))
    print("GENDER DISTRIBUTION:", gender_map)
    print("RATIO FEMALES:", np.round(gender_map[1]/sum(patient_included),4))
    print('RACE DISTRIBUTION:', race_map)
    print("AGE DISTRIBUTION:", age_patients)



def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
