#!/bin/bash
#
#SBATCH --job-name=lstm_ins
#SBATCH --time=2200:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=60G
#SBATCH --chdir ./outputs 

cd /share/pi/boussard/eroosli_work/benchmarking

echo TRAINING THE CHANNEL-WISE LSTM with additional data on insurance
python -um models.ihm.main --network models/keras_models/channel_wise_lstms.py --data data/mimic/aug/mortality --mask_demographics "Gender" "Ethnicity" --dim 8 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --mode train --size_coef 4.0 --epochs 100
