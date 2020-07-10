# +-------------------------------------------------------------------------------------------------+
# | configs.py: contains parameters shared by different analysis files                              |
# |                                                                                                 |
# | Eliane Röösli (2020)                                                                            |
# +-------------------------------------------------------------------------------------------------+


labeldict = {'Overall': 'Total', 
             'Public': 'Public', 'Private': 'Private', 'Medicare': 'Medicare', 'Medicaid': 'Medicaid', 
             'M': 'Male', 'F': 'Female', 
              'ASIAN': 'Asian', 'HISPANIC': 'Hispanic', 'BLACK': 'Black', 'WHITE': 'White'}

colors = {'grid':'#fafafa', 'frame': '#c4c3c2', 
          'val': '#7d3737', 'val_base': '#d4c3c3', 'train': '#3c6350', 'train_base': '#b4d4c5', 'point': '#696969',
         'cint': ['#3c6350', '#7d3737']}

charlson_curves = ['#204e6e', '#056924', '#44ad65', '#138b91']

groups_excluded = ['Selfpay', 'Self Pay', 'Government', 'OTHER', 'Other']

is_public_map = {"1": 1, "2": 1, "3": 1, "4": 0, "5": 2, "0": 2}

two_curves = ['#737373', '#bababa']

paths = {
    'output': 'models/outputs',
    'cv': 'plots/confvals',
    'cb': 'plots/calibration'
}

metrics = {'AUC of ROC': {'name': 'AUROC', 'ax': 0},
           'AUC of PRC': {'name': 'AUPRC', 'ax': 0},
           'Accuracy': {'name': 'Accuracy', 'ax': 1},
           'Precision0': {'name': 'Precision Non-Event', 'ax': 2},
           'Precision1': {'name': 'Precision Event', 'ax': 2},
           'Recall0': {'name': 'Recall Non-Event', 'ax': 3}, 
           'Recall1': {'name': 'Recall Event', 'ax': 3}}
