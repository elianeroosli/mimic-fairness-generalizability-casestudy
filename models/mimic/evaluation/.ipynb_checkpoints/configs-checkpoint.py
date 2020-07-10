labeldict = {'Overall': 'Total', 
             'Public': 'Public Insurance', 'Private': 'Private Insurance', 'Medicare': 'Medicare', 'Medicaid': 'Medicaid', 
             'M': 'Male', 'F': 'Female', 
              'ASIAN': 'Asian', 'HISPANIC': 'Hispanic/Latino', 'BLACK': 'Black', 'WHITE': 'White'}

colors = {'grid':'#fafafa', 'frame': '#c4c3c2', 
          'val': '#7d3737', 'val_base': '#d4c3c3', 'train': '#3c6350', 'train_base': '#b4d4c5', 'point': '#696969',
         'cint': ['#3c6350', '#7d3737']}

groups_excluded = ['Selfpay', 'Self Pay', 'Government', 'OTHER']

is_public_map = {"1": 1, "2": 1, "3": 1, "4": 0, "5": 2}

intervals = ['#3c6350', '#7d3737']