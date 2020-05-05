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


