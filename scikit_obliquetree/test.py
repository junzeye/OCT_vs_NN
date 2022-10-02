from HHCART import HouseHolderCART
from segmentor import MeanSegmentor, Gini
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

x_df = pd.read_csv('x.csv', header=None, delimiter=',')
y_df = pd.read_csv('y.csv', header=None, delimiter=',')
x = np.array(x_df)
y = np.reshape(np.array(y_df), np.array(y_df).size)

# random_state: tried 5
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

sgmtr = MeanSegmentor()
tree = HouseHolderCART(impurity = Gini(), segmentor = sgmtr, max_depth = 10)

tree.fit(x_train, y_train)
print(tree.score(x_test, y_test))