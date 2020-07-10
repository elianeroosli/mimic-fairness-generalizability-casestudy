#!/bin/bash
#
#SBATCH --job-name=lstm_eth
#SBATCH --time=2200:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=60G
#SBATCH --chdir ./outputs 

cd /share/pi/boussard/eroosli_work/benchmarking

echo TRAINING THE CHANNEL-WISE LSTM with additional data on ethnicity 
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/channel_wise_lstms.py --data data/aug/mortality --mask_demographics "Gender" "Insurance" --dim 8 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --mode train --size_coef 4.0 --epochs 100

