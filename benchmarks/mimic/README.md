# MIMIC-III v1.4

## Step-by-step instruction


**1. Get data**

Access to MIMIC-III must be requested through an accreditation process on
`https://mimic.physionet.org/`. Then, the csv files have to be downloaded and
stored at `path_mimic`.
   
**2. Get code**
    
Clone this repo to your desired location.
    
**3. Extract subjects**

We start by extracting the subjects based on the following exclusion criteria:
* ICU transfers
* 2+ ICU stays per admission
* pediatric patients (<18 years)

The first command places you in the directory. It then takes MIMIC-III csv files from 
`path_mimic`, generates one directory per `SUBJECT_ID` in `data` and writes:
* ICU stay information to `data/{SUBJECT_ID}/stays.csv`
* diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`
* events to `data/{SUBJECT_ID}/events.csv`

Note that there is a flag that allows to specify whether additional demographic
variables should be included in the ICU stay information.

       python -m benchmarks.scripts.extract_subjects path_mimic data {--add_demographics}

**4. Clean events data** 

Next, the events data gets cleaned by excluding:
* events with missing HADM_ID
* events with invalid HADM_ID or ICUSTAY_ID

It also retrieves missing ICUSTAY_ID by making use of the HADM_ID. This results 
in the total exclusion of roughly 20% of all events.

       python -m benchmarks.mimic.scripts.validate_events data


**5. Break up per-subject data into separate episodes**

A timeseries of events pertaining to the 17 selected physiological variables
is stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` while episode-level information 
(patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are 
stored in ```{SUBJECT_ID}/episode{#}.csv```. Following the choice in step 3, the timeseries 
can be augmented by demographic information (gender, ethnicity and insurance type) 
if setting the --add_demographics flag to true again.

       python -m benchmarks.mimic.scripts.extract_episodes_from_subjects data {--add_demographics}

**6. Split into training and testing sets**

The split is based on using 85% of the data for training and 15% for testing. 
The seed is fixed to assure that the train/test split remains the same.

       python -m benchmarks.mimic.scripts.split_train_test data
	
**7. Generate task-specific datasets**

These datasets can then be used to train and test models for the specific task, in
our case in-hospital-mortality:

       python -m benchmarks.mimic.scripts.create_ihm data data/mortality/


Before using a model, the training set has to be further split into training
and validation sets as following:
    
        python -m benchmarks.mimic.scripts.split_train_val data/mortality
    
    
## Resulting database

After following all steps, there will be a directory `data/{task}` for each created benchmark task, i.e. mortality in our case.
These directories have two sub-directories: `train` and `test`.
Each of them contains csv files containing events data from ICU stays and one file named `listfile.csv`, 
which lists all samples in that particular set: `icu_stay, period_length, label(s)`. It hence summarizes
the ICUSTAY_ID, the events window (first `period_length` hours of the stay) and the target labels.
The `period_length` for the in-hospital mortality prediction task is always 48 hours, so it is not listed in the corresponding listfiles.

