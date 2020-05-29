
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# handeling missing data


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN" ,strategy="mean", axis = 0)
impiter = imputer.fit(X[:,1:3])

X[:, 1:3] = imputer.transform(X[:,1:3])





