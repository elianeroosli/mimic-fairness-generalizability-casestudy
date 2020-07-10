# +-------------------------------------------------------------------------------------------------+
# | utils.py: helper functions for scripts                                                          |
# |                                                                                                 |
# | Eliane Röösli (2020)                                                                            |
# +-------------------------------------------------------------------------------------------------+

import pandas as pd
import os

def create_listfile(data_path, stays):
    header = "stay,y_true,comorb_90,comorb_180"
    with open(os.path.join(data_path, 'listfile.csv'), 'w') as listfile, open(os.path.join(data_path, 'emptylistfile.csv'), 'w') as empty_listfile:
        # initialize listfiles
        listfile.write(header)
        empty_listfile.write(header)
        
        for idx, stay in stays.iterrows():
            # stay info
            pat_deid = stay['pat_deid']
            stay_id = stay['stay_id']
            ihm = stay['ihm']
            comorb1 = stay['comorb1']
            comorb2 = stay['comorb2']
            filename = str(pat_deid) + "_episode" + str(stay_id) + "_timeseries.csv"
            
            # load csv file and check if it's empty or not
            path = os.path.join(data_path, filename)
            if os.path.exists(path):
                listfile.write("\n" + filename + "," + str(ihm) + "," + str(comorb1) + "," + str(comorb2))
            else:
                empty_listfile.write("\n" + filename + "," + str(ihm) + "," + str(comorb1) + "," + str(comorb2))

def add_listfiles(path):
    for partition in ['train', 'test']:
        csv_list = os.listdir(os.path.join(path, partition))
        csv_list = [file for file in csv_list if file.endswith(".csv")]
        total_stays = pd.read_csv(os.path.join(path, 'listfile.csv'))
        to_keep = [True if stay_csv in csv_list else False for stay_csv in total_stays['stay']]
        total_stays[to_keep].to_csv(os.path.join(path, partition, 'listfile.csv'), index=False)        