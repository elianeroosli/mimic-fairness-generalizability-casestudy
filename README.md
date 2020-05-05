# MIMIC III Model Fairness and Generalizability

This repository will contain all code related to:

- Generating benchmark datasets:
    - Baseline MIMIC III cohort
    - Augmented MIMIC III cohort (additional demographical data)
    - Corresponding STARR cohort
- Training and testing procedures to predict: in-hospital mortality (IHM)
- Analysis tools for:
    - performance evaluation
    - fairness: anti-classification, classification parity, and calibration
    - possibly additional tools for generalizability and representativity


## Benchmark building pipeline

This data preparation pipeline is adapted from Harutyunyan et al.'s article on 
[Multitask learning and benchmarking with clinical time series data](https://www.nature.com/articles/s41597-019-0103-9), 
with the code available [here](https://github.com/YerevaNN/mimic3-benchmarks).


### MIMIC-III (v1.4) cohort
---
#### Step-by-step instruction

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
    
    
#### Resulting database

After following all steps, there will be a directory `data/{task}` for each created benchmark task.
These directories have two sub-directories: `train` and `test`.
Each of them contains csv files containing events data from ICU stays and one file named `listfile.csv`, 
which lists all samples in that particular set: `icu_stay, period_length, label(s)`. It hence summarizes
the ICUSTAY_ID, the events window (first `period_length` hours of the stay) and the target labels.
The `period_length` for the in-hospital mortality prediction task is always 48 hours, so it is not listed in the corresponding listfiles.


### STARR_DE cohort
------

#### Step-by-step instruction

#### Resulting database

## Benchmark model training

Harutyunyan's paper looks at four clinical prediction tasks for ICU patients: 
in-hospital mortality, decompensation, length-of-stay and phenotyping. 
Five different baseline models for each of the four main tasks have been provided:

- Linear/logistic regression
- Standard LSTM
- Standard LSTM + deep supervision
- Channel-wise LSTM
- Channel-wise LSTM + deep supervision

In addition, they have also developped multitasking models that aim to learn all four
prediction tasks simultaneously. For the frame of this project however, we focus
on the modeling of in-hospital mortality. The best-performing
model for this task was reported to be the `simple channel-wise LSTM`. Hence, we focus
on analysing this specific model on bias, demographic fairness and generalizability.


### General step-by-step instructions

The code for the LSTM-based models can be found in the mimic3models/keras_models directory.
The `main.py` files to train the models are situated in their respective mimic3models/{task} directories.
As these models take relatively long to train, it is advised to submit a job file to SLURM on NERO.
These can be found in the `slurm_jobs` directory. The following steps have to be followed to successfully
submit a job:

**1. Activate environment**

The models make use of keras, which is not part of the base environment. Therefore, another environment
containing keras has to be activated:

        source activate /share/pi/boussard/envs/eroosli_env
        
**2. Go to slurm_job directory**

        cd /share/pi/boussard/eroosli_work/benchmarking/slurm_jobs
        

**3. Select corresponding shell script file**

There are two options for the channel-wise LSTM model to predict in-hospital mortality:
Use the basic benchmark dataset or the augmented dataset including three additional 
demographic variables (gender, ethnicity and insurance). The corresponding .sh files
are named `main.py` and `main_aug.py` respectively.
    
**4. Update and validate shell script file**

Go over all code chunks in the .sh file to make sure they fit with your needs and file organization:

- main script file: mimic3models.in_hospital_mortality.main_aug
- model: mimic3models/keras_models/channel_wise_lstms_aug.py
- data: data/aug/full_aug/mortality 
- additional parameters: --dim 8 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --mode train --size_coef 4.0 --epochs 50

**5. Submit job to SLURM**

        sbatch filename.sh
        
**6. Check status of job**

        squeue -u SUNETID
        
**7. Analyse output**

The output file corresponding to the submitted job can be found in the `slurm_jobs/outputs` directory
and is uniquely identified by its job ID.


## Benchmark model testing



talk about which epoch is best to use for testing etc.


