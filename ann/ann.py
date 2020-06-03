import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path = "C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\ann\\"
# Importing the dataset
dataset = pd.read_csv(path+'Churn_Modelling.csv')


X = dataset.iloc[:, 3:13].values #3-12 
y = dataset.iloc[:, 4].values



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


# part 2 - now let's make the ANN!

# importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# initialising the ANN
classifier = Sequential()
# step 1 : initialize the weight randomly small numbers close to 0
#   -- this will be done by Dense layrt
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))


# step 2 : input the first observation of your dataset in the input layer ,each feature in one input node.

classifier.add(Dense(output_dim = 6 , init = 'uniform' , activation= 'relu'))

# step 3 : forward propagation : 

clasifier.add(Dense(output_dim = 1 , init= 'uniform',activation = 'sigmoid'))








