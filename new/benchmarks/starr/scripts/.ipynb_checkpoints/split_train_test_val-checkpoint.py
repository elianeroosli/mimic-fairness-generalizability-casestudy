# +-------------------------------------------------------------------------------------------------+
# | split_train_test_val.py: partition data for model training and evaluation                       |
# |                                                                                                 |
# | Eliane Röösli (2020)                                                                            |
# +-------------------------------------------------------------------------------------------------+

import argparse
import shutil
import os
import numpy as np
from sklearn.model_selection import train_test_split
from benchmarks.starr.utils import add_listfiles


# move data to according partition
def move_to_partition(path, stays, partition, move=True):
    if not os.path.exists(os.path.join(path, partition)):
        os.mkdir(os.path.join(path, partition))
    for stay in stays:
        src = os.path.join(path, stay)
        dest = os.path.join(path, partition, stay)
        if move:
            shutil.move(src, dest)
        else:
            shutil.copy(src, dest)
        
# save data in listfile for a given partition
def save_listfile(path, partition, header, X_data, y_data, c1_data, c2_data):
    with open(os.path.join(path, partition + '_listfile.csv'), 'w') as listfile:
        listfile.write(header)
        for stay in range(len(X_data)):
            listfile.write(str(X_data[stay]) + "," + str(y_data[stay]) + "," + str(c1_data[stay]) + "," + str(c2_data[stay]) + "\n")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train, test and validation sets.")
    parser.add_argument('dataset_dir', type=str, help='Path to the directory which contains the dataset')
    args, _ = parser.parse_known_args()
    
    # read data
    with open(os.path.join(args.dataset_dir, 'listfile.csv')) as listfile:
        lines = listfile.readlines()
        header = lines[0]
        lines = lines[1:]
        X_data = []
        y_data = []
        c90_data = []
        c180_data = []
        
        for line in lines:
            x, y, c90, c180 = line.split(',')
            X_data.append(x)
            y = str(y).replace('\n', '')
            y_data.append(int(y))
            c90 = str(c90).replace('\n', '')
            c90_data.append(float(c90))
            c180 = str(c180).replace('\n', '')
            c180_data.append(float(c180))
        
        # split into train and test batches
        X_train, X_test, y_train, y_test, c90_train, c90_test, c180_train, c180_test = train_test_split(X_data, y_data, c90_data, c180_data, \
                                                                                                        test_size=0.15, shuffle=True, stratify=y_data, random_state=2020)

        # keep copy of full data directory         
        move_to_partition(args.dataset_dir, X_data, "full", move=False)
        
        # split train and test
        move_to_partition(args.dataset_dir, X_train, "train")
        move_to_partition(args.dataset_dir, X_test, "test")

        # split train data further into train and validation data
        X_train, X_val, y_train, y_val, c90_train, c90_val, c180_train, c180_val = train_test_split(X_train, y_train, c90_train, c180_train, \
                                                                                                    test_size=0.15, shuffle=True, stratify=y_train, random_state=2020)

        # check whether splitting has been done correctly
        assert len(X_train) + len(X_val) + len(X_test) == len(X_data)
        assert len(y_train) + len(y_val) + len(y_test) == len(y_data)

        # save partition information in csv files
        save_listfile(args.dataset_dir, 'train', header, X_train, y_train, c90_train, c180_train)
        save_listfile(args.dataset_dir, 'test', header, X_test, y_test, c90_test, c180_test)
        save_listfile(args.dataset_dir, 'val', header, X_val, y_val, c90_val, c180_val)
        add_listfiles(args.dataset_dir)
            

