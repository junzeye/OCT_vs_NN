import time, itertools, sys, warnings, os
warnings.simplefilter(action='ignore', category=FutureWarning)
from os import path
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from tree import optimalDecisionTreeClassifier
import Ch2_framework.dataset as dataset
import tree as miptree

## Optimal Classification Tree 

timelimit = 2700 # time limit
datasets = ['CTG_width']
alpha = [0.001, 0.01, 0.1]
depth = [4,6,8]
min_samples_split = 2 # minimum number of leaves in a node
hidden_dims = [20, 40, 60, 80, 100, 120]
seeds = [37, 42, 53] # use of specified random states for experiment reproducibility

train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25

# create or load previously computed results
res_sk = pd.DataFrame(columns=['instance', 'depth', 'seed', 'train_acc', 'val_acc', 'test_acc', 'train_time'])
if path.isfile('./res/oct.csv'):
    res_oct = pd.read_csv('./res/oct.csv')
else:
    res_oct = pd.DataFrame(columns=['instance', 'depth', 'alpha', 'seed', 
                                    'train_acc', 'val_acc', 'test_acc', 'train_time', 'gap'])

for data, hidden_dim, d, s in itertools.product(datasets, hidden_dims, depth, seeds): # simplifies nested for loops
    # load data
    x, y = dataset.loadData(data, hidden_dim)
    # data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_ratio, random_state=s)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, 
                                                    test_size=test_ratio/(test_ratio+val_ratio), random_state=s)

    for a in alpha:
        # oct
        row = res_oct[(res_oct['instance'] == data) & (res_oct['depth'] == d) & 
                    (res_oct['alpha'] == a) & (res_oct['seed'] == s) & (res_oct['gap'] <= 0.02)]
        if len(row): # if the specific oct is already trained and optimized up to a degree, we directly print the previously recorded results
            print(data, 'oct-d{}-a{}'.format(row['depth'].values[0],row['alpha'].values[0]),
                'train acc:', row['train_acc'].values[0], 'val acc:', row['val_acc'].values[0],
                'gap:', row['gap'].values[0])
        else: # tree has not been trained, so train
            octree = optimalDecisionTreeClassifier(max_depth=d, min_samples_split=min_samples_split, alpha=a, warmstart=True,
                                                        timelimit=timelimit, output=True)
            orig_stdout = sys.stdout
            filepath = f'synthetic_tests/{data}/logs_dim={hidden_dim}_d={d}_seed={s}_alpha={a}_minLeaf={min_samples_split}.txt'
            os.makedirs(os.path.dirname(filepath), exist_ok = True)
            with open(filepath, 'w') as f:
                sys.stdout = f
                tick = time.time()
                
                octree.fit(x_train, y_train)
                
                tock = time.time()
            sys.stdout = orig_stdout
            
            train_time = tock - tick
            train_acc = accuracy_score(y_train, octree.predict(x_train))
            val_acc = accuracy_score(y_val, octree.predict(x_val))
            test_acc = accuracy_score(y_test, octree.predict(x_test))
            row = {'instance':data, 'depth':d, 'alpha':a, 'seed':s, 'train_acc':train_acc, 'val_acc':val_acc,
                'test_acc':test_acc, 'train_time':train_time, 'gap':octree.optgap}
            res_oct = res_oct.append(row, ignore_index=True)
            res_oct.to_csv('./res/oct.csv', index=False)
            print(data, 'oct-d{}-a{}'.format(d,a), 
                'train acc:', train_acc, 'val acc:', val_acc, 'gap:', octree.optgap)

