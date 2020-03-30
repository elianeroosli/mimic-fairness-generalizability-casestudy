# MIMIC III Model Fairness and Generalizability

This repository will contain all code related to:

- Generating benchmark datasets:
    - Baseline MIMIC III cohort
    - Augmented MIMIC III cohort (additional demographical data)
    - Corresponding STARR cohort
- Training models to predict:
    - In-hospital mortality (IHM)
    - Decompensation (DEC)
- Analysis tools for fairness and generalizability


## Benchmark building pipeline

This data preparation pipeline is adapted from Harytyunyan et al.'s article on 
[Multitask learning and benchmarking with clinical time series data](https://www.nature.com/articles/s41597-019-0103-9), 
with the code available [here](https://github.com/YerevaNN/mimic3-benchmarks).

### Step-by-step instruction

**1. Get data**

Access to MIMIC-III must be requested through an accreditation process on
`https://mimic.physionet.org/`. Then, the csv files have to be downloaded and
stored at path_mimic.
   
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

       cd mimic3-benchmarks/
       python -m mimic3benchmark.scripts.extract_subjects path_mimic data --add_demographics {True/False}

**4. Clean events data** 

Next, the events data gets cleaned by excluding:
* events with missing HADM_ID
* events with invalid HADM_ID or ICUSTAY_ID

It also retrieves missing ICUSTAY_ID by making use of the HADM_ID. This results 
in the total exclusion of roughly 20% of all events.

       python -m mimic3benchmark.scripts.validate_events data


**5. Break up per-subject data into separate episodes**

A timeseries of events pertaining to the 17 selected physiological variables
is stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` while episode-level information 
(patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are 
stored in ```{SUBJECT_ID}/episode{#}.csv```. Following the choice in step 3, the timeseries 
can be augmented by demographic information (gender, ethnicity and insurance type) 
if setting the --add_demographics flag to true again.

       python -m mimic3benchmark.scripts.extract_episodes_from_subjects data --add_demographics {True/False}

**6. Split into training and testing sets**

The split is based on using 85% of the data for training and 15% for testing. 
The seed is fixed to assure that the train/test split remains the same.

       python -m mimic3benchmark.scripts.split_train_and_test data
	
**7. Generate task-specific datasets**

These datasets can then be used to train and test models for the specific task. 
Each command is independent.

       python -m mimic3benchmark.scripts.create_in_hospital_mortality data data/mortality/
       python -m mimic3benchmark.scripts.create_decompensation data data/decompensation/

Before using a model, the training set has to be further split into training
and validation sets as following:
    
        python -m mimic3models.split_train_val data/{task}
    
    
### Resulting database

After following all steps, there will be a directory `data/{task}` for each created benchmark task.
These directories have two sub-directories: `train` and `test`.
Each of them contains csv files containing events data from ICU stays and one file named `listfile.csv`, 
which lists all samples in that particular set: `icu_stay, period_length, label(s)`. It hence summarizes
the ICUSTAY_ID, the events window (first `period_length` hours of the stay) and the target labels.
The `period_length` for the in-hospital mortality prediction task is always 48 hours, so it is not listed in the corresponding listfiles.


## Benchmark model testing

Harytyunyan's paper develops seven different models for four clinical prediction tasks for ICU patients: 
in-hospital mortality, decompensation, length-of-stay and phenotyping. 