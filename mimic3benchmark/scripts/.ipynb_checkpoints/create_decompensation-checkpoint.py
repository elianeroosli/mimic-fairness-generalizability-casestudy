from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import random
random.seed(49297)


def process_partition(args, partition, sample_rate=1.0, shortest_length=4.0,
                      eps=1e-6, future_time_interval=24.0):

    # prepare output directory
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    
    print('PARTITION FOR:', partition)
    nb_patients = len(patients)
    patient_included = [0]*nb_patients
    stay_included = [0]*nb_patients
    
    patient_included_test = [0]*nb_patients
    stay_included_test = [0]*nb_patients
    
     # start with root cohort
    for (patient_index, patient) in enumerate(patients):
        
        sys.stdout.write('\rSUBJECT {0} of {1}...'.format(patient_index+1, nb_patients))
        
        # get patient timeseries event data from folder
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        stays_df = pd.read_csv(os.path.join(patient_folder, "stays.csv"))

        # for each episode during an admission
        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                # get diagnosis data from the non timeseries csv file
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    continue

                # (1) exclude episodes with missing LOS
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    #print("(length of stay is missing)", patient, ts_filename)
                    continue
                
                # determine the lived-time while being hospitalized (1e18 if not dead)
                stay = stays_df[stays_df.ICUSTAY_ID == label_df.iloc[0]['Icustay']]
                deathtime = stay['DEATHTIME'].iloc[0]
                intime = stay['INTIME'].iloc[0]
                if pd.isnull(deathtime):
                    lived_time = 1e18
                else:
                    lived_time = (datetime.strptime(deathtime, "%Y-%m-%d %H:%M:%S") -
                                  datetime.strptime(intime, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0

                # get events during whole stay for a given episode
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]
                event_times = [t for t in event_times
                               if -eps < t < los + eps]

                # (2) exclude episodes with no measurements in ICU
                if len(ts_lines) == 0:
                    #print("(no events in ICU) ", patient, ts_filename)
                    continue

                # create array of all sample times until end of stay or death given by sample_rate
                sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate)

                # exclude all sample times lower than shortest_length (default=4.0)
                sample_times = list(filter(lambda x: x > shortest_length, sample_times))

                # At least one measurement
                sample_times = list(filter(lambda x: x > event_times[0], sample_times))

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                mortality = int(label_df.iloc[0]["Mortality"])
                
                stay_included_test[patient_index] += 1
                patient_included_test[patient_index] = 1
                
                if sample_times != []:
                    stay_included[patient_index] += 1
                    patient_included[patient_index] = 1
                        
                # (3) if sample_times is empty (samples shorter than 4h), not added to xty_triples
                for t in sample_times:
                    if mortality == 0:
                        cur_mortality = 0
                    else:
                        cur_mortality = int(lived_time - t < future_time_interval)
                    xty_triples.append((output_ts_filename, t, cur_mortality))
                    # at least one episode has been retained for the given patient
                    

    print("\n TOTAL NUMBER SAMPLES:", len(xty_triples))
    print("TOTAL NUMBER PATIENTS:", sum(patient_included))
    print("TOTAL NUMBER STAYS:", sum(stay_included))
    print("TOTAL NUMBER PATIENTS TEST:", sum(patient_included_test))
    print("TOTAL NUMBER STAYS TEST:", sum(stay_included_test))
    if partition == "train":
        random.shuffle(xty_triples)
    if partition == "test":
        xty_triples = sorted(xty_triples)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,y_true\n')
        for (x, t, y) in xty_triples:
            listfile.write('{},{:.6f},{:d}\n'.format(x, t, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for decompensation prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
