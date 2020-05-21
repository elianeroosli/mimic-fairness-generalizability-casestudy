import pandas as pd
import os

def create_listfile(data_path, stays):
    header = "stay,y_true"
    header_empty = "stay,y_true,hosp_in,hosp_out"
    with open(os.path.join(data_path, 'listfile.csv'), 'w') as listfile, open(os.path.join(data_path, 'emptylistfile.csv'), 'w') as emptylistfile:
        # initialize listfiles
        listfile.write(header)
        emptylistfile.write(header_empty)
        
        for idx, stay in stays.iterrows():
            # stay info
            pat_deid = stay['pat_deid']
            stay_id = stay['stay_id']
            ihm = stay['ihm']
            hosp_in = stay['hosp_in']
            hosp_out = stay['hosp_out']
            filename = str(pat_deid) + "_episode" + str(stay_id) + "_timeseries.csv"
            
            # load csv file and check if it's empty or not
            df = pd.read_csv(os.path.join(data_path, filename))
            if df.empty:
                data = ",".join([filename, str(ihm), hosp_in, hosp_out])
                emptylistfile.write("\n" + data)
            else:
                listfile.write("\n" + filename + "," + str(ihm))