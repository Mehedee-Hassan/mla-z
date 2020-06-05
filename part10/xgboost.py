# XGBoostt

# Install xgboost following the instructions on this link 
import xgboost

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path = "C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\ann\\"
# Importing the dataset
dataset = pd.read_csv(path+'Churn_Modelling.csv')


X = dataset.iloc[:, 3:13].values #3-12 
y = dataset.iloc[:, 13].values



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])



onehotencoder = OneHotEncoder(categorical_features=[1])
X= onehotencoder.fit_transform(X).toarray()

X = X[:,1:]
# encoding the dependent variable 
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

# the implementatio is very weak
# binary parameter for the classifier 
# training set 



#section 2
y_pred =classifier.predict(X_test)



# making a confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# applying k-fold cross valiedation 
from sklearn.model_selection import cross_val_score

# return accuracy for 10 combination    
# this will get 10 accuracies   
# n_jobs : if you are working with a very hight amount of data 
# n_jobs = -1 will use all the cpy of your pc to calculate the result 
accuracies = cross_val_score(estimator= classifier,X = X_train, y= y_train,cv = 10,n_jobs = -1)

avg_accuracies = accuracies.mean()
std_accuracies = accuracies.std()





