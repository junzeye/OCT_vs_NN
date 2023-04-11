# To allow for proper module importing and file referencing, please set the pwd 
# by cd to 'Senior_Thesis_Code/' on command line.

import time, itertools, sys, warnings, os, json
warnings.simplefilter(action='ignore', category=FutureWarning)
from os import path
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
import Ch2_framework.dataset as dataset
from Ch4_oblique.scikit_obliquetree.HHCART import HouseHolderCART
from Ch4_oblique.scikit_obliquetree.segmentor import MeanSegmentor, TotalSegmentor, Gini

start_time = time.time()

MAX_TIME = 60 * 60 * 64 # safe exit time in seconds
max_depths = list(range(2,31,2)) # [2,4,6,8,,...,30]
min_sample_leaves = [2,4,8,16,32]
hidden_dims = [20,40,60,80,100,120]

random_states = [0,1,2,3]
cv_random_states = {0:4, 1:5, 2:6, 3:7} # random_states when doing cross-validation
train_ratio = 0.8
test_ratio = 0.2
CSV_FILEPATH = 'res/oblique.csv'
COLUMNS= ['NN_dim','CART_depth','seed','min_samples_leaf','train_acc','test_acc','train_time']
data = 'CTG_width'

# create or load previously computed results
if path.isfile(CSV_FILEPATH):
    res_oblique = pd.read_csv(CSV_FILEPATH)
else:
    res_oblique = pd.DataFrame(columns=COLUMNS)

for hidden_dim in hidden_dims:
    # load data
    x, y = dataset.loadData(data, hidden_dim)

    for d, s in itertools.product(max_depths, random_states):
        # tree has not been trained, so train
        # data splitting
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_ratio, 
                                        random_state=s)
        scores_by_leaves = [0] * len(min_sample_leaves)

        tick = time.time()

        # use cross validation to select the optimal number of minimum samples per leaf
        for index, l in enumerate(min_sample_leaves):
            sgmtr = TotalSegmentor()
            cv = StratifiedKFold(shuffle = True, random_state = cv_random_states[s]) # default 5-fold cv
            HHTree = HouseHolderCART(impurity = Gini(), segmentor = sgmtr, max_depth = d, min_samples = l)
            scores = cross_val_score(HHTree, x_train, y_train, cv = cv)
            scores_by_leaves[index] = sum(scores) / len(scores)
        
        train_time = time.time() - tick

        best_index = np.argmax(scores_by_leaves) # min_sample_leaf with the best validation accuracy
    
        HHTree = HouseHolderCART(impurity = Gini(), segmentor = sgmtr, max_depth = d, 
                                    min_samples = min_sample_leaves[best_index])
        HHTree.fit(x_train, y_train)
        train_score = accuracy_score(y_train, HHTree.predict(x_train))
        test_score = accuracy_score(y_test, HHTree.predict(x_test))
        
        row = {'NN_dim': hidden_dim, 'CART_depth': d, 'seed': s, 'min_samples_leaf': min_sample_leaves[best_index], 
                'train_acc': train_score, 'test_acc': test_score, 'train_time': train_time}
        res_oblique = res_oblique.append(pd.DataFrame.from_records([row]), ignore_index = True)
        
        time_elapsed = time.time() - start_time
        if time_elapsed > MAX_TIME:
            break

    res_oblique.sort_values(['NN_dim', 'CART_depth', 'test_acc'])
    res_oblique.to_csv('res/oblique.csv', index = False)