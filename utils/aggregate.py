import json, sys, os
import pandas as pd

# same as the one in experiment_instance.py
COLUMNS= ['instance', 'hidden_dim', 'depth', 'alpha', 'seed', 'train_acc', 'val_acc', 
        'test_acc', 'train_time', 'gap']
records = pd.DataFrame(columns= COLUMNS)

dirname = '../synthetic_tests/CTG_width'

# aggregate all the .out files into a single dataframe
for file in os.listdir(dirname):
    if '.out' in file:
        dic = json.load(os.path.join(dirname, file))
        records = pd.concat([records, pd.DataFrame.from_records(dic)], 
                            ignore_index = True)
        records.sort_values(['hidden_dim','train_time', 'depth', 'alpha', 'seed'], 
                            inplace = True)

records.to_csv('../res/oct_vary_width.csv')