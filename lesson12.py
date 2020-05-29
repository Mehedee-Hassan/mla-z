# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:03:42 2020

@author: mehedee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.preprocessing import LabelEncoder ,OneHotEncoder

dataset = pd.read_csv('C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\Data.csv')

labelEncoder_X = LabelEncoder()
# to apply
X[:,0] = labelEncoder_X.fit_transform(X[:,0])


onehotEncoder_X = OneHotEncoder(categorical_features=[0])
X = onehotEncoder_X.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(Y)



from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0) 





#feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

























