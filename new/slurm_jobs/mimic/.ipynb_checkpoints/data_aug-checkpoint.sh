#!/bin/bash
#
#SBATCH --job-name=data_aug
#SBATCH --time=1200:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=50G
#SBATCH --chdir ./outputs

cd /share/pi/boussard/eroosli_work/benchmarking


echo STEP 1: EXTRACT SUBJECTS
python -um benchmarks.mimic.scripts.extract_subjects ../mimic-iii-clinical-database-1.4 data/mimic/aug --add_demographics 


echo STEP 2: VALIDATE EVENTS
python -um benchmarks.mimic.scripts.validate_events data/mimic/aug


echo STEP 3: EXTRACT EPISODES FROM SUBJECTS
python -um benchmarks.mimic.scripts.extract_episodes_from_subjects data/mimic/aug --add_demographics 


echo STEP 4: SPLIT TRAIN AND TEST
python -um benchmarks.mimic.scripts.split_train_test data/mimic/aug


echo STEP 5: CREATING IHM DATA
python -um benchmarks.mimic.scripts.create_ihm data/mimic/aug data/mimic/aug/mortality


echo STEP 6: SPLIT TRAIN AND VALIDATION DATA
python -um benchmarks.mimic.scripts.split_train_val data/mimic/aug


