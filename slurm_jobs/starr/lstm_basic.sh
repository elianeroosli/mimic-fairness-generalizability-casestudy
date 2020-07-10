#!/bin/bash
#
#SBATCH --job-name=lstm_basic
#SBATCH --time=2000:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40G
#SBATCH --chdir ./outputs 

cd /share/pi/boussard/eroosli_work/benchmarking

echo TRAINING THE CHANNEL-WISE LSTM without additional data

python -um models.ihm.main --network models/keras_models/channel_wise_lstms.py --data data/starr/ihm --data_name 'starr' --output_dir models/outputs/starr --mask_demographics "Ethnicity" "Gender" "Insurance" --dim 8 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --mode train --size_coef 4.0 --epochs 100

