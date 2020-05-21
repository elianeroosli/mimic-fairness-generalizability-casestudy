import argparse
import shutil
import os
from sklearn.model_selection import train_test_split


# move data to according partition
def move_to_partition(path, stays, partition):
    if not os.path.exists(os.path.join(path, partition)):
        os.mkdir(os.path.join(path, partition))
    for stay in stays:
        src = os.path.join(path, stay)
        dest = os.path.join(path, partition, stay)
        shutil.move(src, dest)
        
# save data in listfile for a given partition
def save_listfile(path, partition, header, X_data, y_data):
    with open(os.path.join(path, partition + '_listfile.csv'), 'w') as listfile:
        listfile.write(header)
        for stay in range(len(X_data)):
            listfile.write("\n" + str(X_data[stay]) + "," + str(y_data[stay]))

    
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
        
        for line in lines:
            x, y = line.split(',')
            X_data.append(x)
            y = str(y).replace('\n', '')
            y_data.append(int(y))
            
        # split into train and test batches
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.15, shuffle=True, stratify=y_data, random_state=2020)

        # split train and test
        move_to_partition(args.dataset_dir, X_train, "train")
        move_to_partition(args.dataset_dir, X_test, "test")

        # split train data further into train and validation data
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, shuffle=True, stratify=y_train, random_state=2020)

        # check whether splitting has been done correctly
        assert len(X_train) + len(X_val) + len(X_test) == len(X_data)
        assert len(y_train) + len(y_val) + len(y_test) == len(y_data)

        # save partition information in csv files
        save_listfile(args.dataset_dir, 'train', header, X_train, y_train)
        save_listfile(args.dataset_dir, 'test', header, X_test, y_test)
        save_listfile(args.dataset_dir, 'val', header, X_val, y_val)
            

