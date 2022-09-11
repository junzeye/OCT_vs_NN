import numpy as np
import pandas as pd
from collections import Counter

for hidden_dim in [20,40,60,80,100,120]:
    filepath_y = f'synthetic_data/vary_width/dim_{hidden_dim}/y.csv'
    data = pd.read_csv(filepath_y)
    y = list(data.iloc[:,0])
    counter = dict(Counter(y))

print(dict(counter))