## Step-by-step instructions: Training

The code for the LSTM-based models can be found in the `keras_models` directory and
the ``main.py`` file to train the models is situated in the `in_hospital_mortality` directory.
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

There are several options for the channel-wise LSTM model to predict in-hospital mortality:
Use the basic benchmark dataset or augmenting it with a selection of demographic variables
choosing from *gender, ethnicity and insurance* data. The corresponding .sh files
are named `lstm_{variables}.py`.
    
**4. Update and validate shell script file**

Go over all code chunks in the .sh file to make sure they fit with your needs and file organization:

- main script file: mimic3models.in_hospital_mortality.main
- model: mimic3models/keras_models/channel_wise_lstms.py
- data: data/aug/mortality 
- demographics: --mask_demographics "Ethnicity" "Gender" "Insurance" (a selection of these three)
- additional parameters: --dim 8 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --mode train --size_coef 4.0 --epochs 50

**5. Submit job to SLURM**

        sbatch filename.sh
        
**6. Check status of job**

        squeue -u SUNETID
        
**7. Analyse output**

The output file corresponding to the submitted job can be found in the `slurm_jobs/outputs` directory
and is uniquely identified by its job ID.


## Step-by-step instructions: Testing


**1. Select best model**

The first step is to find the best model epoch based on the validation AUC-ROC score. The above training procedure 
has stored a .csv file in the `keras_logs` directory named in the following fashion:


It contains all the relevant performance metrics on both train and validation data for each training epoch.
The epoch associated with the highest `val_auroc` should be chosen as the final model to be tested.


**2. Predictions on test data**

In a Jupyter notebook, the following script is then run:

        %run models/mimic3/in_hospital_mortality/main.py

It takes a number of parameters to specify the testing procedure:

        --network models/mimic3/keras_models/channel_wise_lstms.py
        --mask_demographics "Ethnicity" "Gender" "Insurance" 
        --data data/aug/mortality 
        --dim 8 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --size_coef 4.0
        --load_state models/mimic3/in_hospital_mortality/keras_states/{date}/k_clstms.{demographics}.epoch{epoch}.state 
        --mode test 
        
**3. Analyse results**

The script has produced several outputs ready for analysis in the `predictions` folder:

- **curves**: .csv files to create ROC and PRC plots, per demographic group and overall
- **metrics**: .csv file with all relevant performance metrics, per demographic group and overall
- **results**: .csv file with the prediction, correct label and demographic data for all tested samples
