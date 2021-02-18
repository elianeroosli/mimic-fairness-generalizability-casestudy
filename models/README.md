# Benchmark models

Harutyunyan's paper looks at four clinical prediction tasks for ICU patients and they have
developed an array of models for each task. For the scope of this project however, we only focus
on the modeling of in-hospital mortality. The best-performing
model for this task was reported to be the `simple channel-wise LSTM`. Hence, we focus
on analyzing this specific model on bias, demographic fairness and generalizability.

## Step-by-step instructions: Training

The code for the LSTM-based models can be found in the `keras_models` directory and
the ``main.py`` file to train the models is situated in the `ihm` directory.

In a Jupyter notebook, the following script is run:

        %run models/ihm/main.py

It takes a number of parameters to specify the training procedure:

        --network models/keras_models/channel_wise_lstms.py
        --data data/{mimic/aug, starr}/mortality 
        --mask_demographics "Ethnicity" "Gender" "Insurance" (a selection of these three)
        --output_dir models/outputs/{starr, mimic}
        --dim 8 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --size_coef 4.0 --epochs 100 {--SMOTE}
        --mode train 


## Step-by-step instructions: Testing


**1. Select best model**

The first step is to find the best model epoch based on the validation AUROC score. The above training procedure 
has stored a .csv file in the `keras_logs` directory which can be fed into the following function:

        models.common_utils.optimal_epoch('{date}.k_clstms.{demographics}.csv', {'mimic', 'starr'})
    
The file contains all the relevant performance metrics on both train and validation data for each training epoch.
The function returns the epoch associated with the highest `val_auroc`. The model associated
with this model epoch should be chosen as the final model to be tested.


**2. Predictions on test data**

In a Jupyter notebook, the following script is then run:

        %run models/ihm/main.py

It takes a number of parameters to specify the testing procedure:

        --network models/keras_models/channel_wise_lstms.py
        --mask_demographics "Ethnicity" "Gender" "Insurance" (matching the training selection)
        --data data/{mimic/aug, starr}/mortality 
        --output_dir models/outputs/{starr, mimic}
        --dim 8 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --size_coef 4.0
        --load_state models/ihm/keras_states/{date}/k_clstms.{demographics}.epoch{epoch}.state 
        --mode test 
        
**3. Analyse results**

The script produces several outputs ready for analysis in the `predictions` folder:

- **curves**: .csv files to create ROC and PRC plots, per demographic group and overall
- **metrics**: .csv file with all relevant performance metrics, per demographic group and overall
- **results**: .csv file with the prediction, correct label and demographic data for all tested samples


## Step-by-step instructions: Evaluation

The evaluation script takes as input the test predictions stored in `results/filename.csv` and computes 
confidence intervals of the relevant performance metrics by bootstrapping. The intervals
are individually computed for all demographic groups and the whole test data. In a Jupyter
notebook, write and execute:

        %run models/evaluation/evaluate_ihm.py 

A number of parameters to specify the evaluation procedure need to be added on the same line:

        - listfile of test data: --test_listfile "data/{mimic/aug, starr}/mortality/test/listfile.csv" 
        - number bootstrapping iterations: --n_iters 10000 
        - whether bootstrapping should be stratified by the outcome label: --stratify 
        - test predictions: "models/ihm/predictions/results/TEST.{date}.k_clstms.{demographics}.csv"

A .json file stores all computed confidence intervals in the `predictions/confvals` directory,
which can then be loaded to perform further analysis and visualizations on them.
