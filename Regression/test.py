import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv('C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\Regression\\test.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
y = y.astype(float)
y = ([y])

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


