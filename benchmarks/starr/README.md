# STARR2019_DE

## Step-by-step instruction

**1. Get access to data**

STARR2019_DE is a private database with from Stanford Health Care with fully identifiable patient data and can therefore not be shared with the public. 

***Note:*** The data has been pulled into STARR2019_DE from different locations, namely via a views 
on SHC_ICU_FLOWSHEET_DE (vitals), from the STRIDE_DATALAKE SHC_FLOWSHEET on BigQuery (vitals) and
tables already present in STARR2019: PAT_MAP_NEW_DE (stays), SHC_ICU_IN_OUT (stays),
SHC_ENCOUNTER_DE (stays) and SHC_LAB_RESULT_DE (labs)
   
**2. Get code**
    
Clone this repo to your desired location.

    
**3. Update params.json file**

To access the Oracle database from Nero, one has to update the username information
in the `sql/params.json` file to create a successful connection.


**4. Fetch data**

The file `fetch_data.py` builds the STAYS, LABS and VITALS table from the SQL queries
in the `sql` folder and loads them into Nero as `.tsv` files.

The tables are built using the same exclusion criteria as detailed in the `mimic`
processing flow. The tables contain the following information:

*STAYS:*
- Patient and stay ID
- Demographic information: age, gender, insurance, ethnicity
- Body anatomy: height, weight
- Dates: admission and discharge times in hospital and ICU
- Target outcome: in-hospital mortality
    
*LABS:*
- Fraction inspired oxygen
- Glucose
- Heart rate
- Oxygen saturation
- pH
    
*VITALS:*
- Blood pressure: diastolic, systolic and mean arterial pressure
- Glasgow coma scale
- Heart rate
- Oxygen saturation
- Respiratory rate
- Temperature
    
The script is run in the following way from the terminal

        cd benchmarks/starr
        python fetch_data.py -p "sql/params.json" [-b] [-l]

Importantly, there are two options one can choose:
- `-b`: triggers the script to first build the tables from scratch again
- `-l`: suppresses the loading of the tables (in case one only wants to rebuild the tables)


**5. Create timeseries**

Then, one single script loads the data, cleans it and creates individual timeseries
stored as `.csv` files for each stay. It is invoked the following way in a Jupyter notebook environment: 

        %run benchmarks/starr/scripts/create_timeseries.py -p "benchmarks/starr/sql/params.json" -v

**6. Split into train, test and validation set**

Finally, the data is split according to the same proportions used in mimic into train, test
and validation sets. 

        %run benchmarks/starr/scripts/split_train_test_val.py "data/starr/ihm"
        
        
