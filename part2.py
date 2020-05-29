
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder



dataset = pd.read_csv('C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\50_Startups.csv')

dataset.head()


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values



from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()



#Avoiding the dummy variabale test 
X = X[:,1:]
# we are ready to make the tarin and test split


X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0) 

#we dont need to take care of feature scaling
# beacude multiple linear regression library will take care of it



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)



y_rped = regressor.predict(X_test)





import statsmodels.api as sm

X = np.append(arr=np.ones((50,1)).astype(int),values = X,axis=1)

#X = np.append(arr = np.ones((50,1)).astype(int), values = X,axis=1) #adding to the column of a matrix
# will appear at the first column


# backward elimination

X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog=y ,exog=X_opt).fit()

regressor_OLS.summary()



X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y ,exog=X_opt).fit()
regressor_OLS.summary()



X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y ,exog=X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y ,exog=X_opt).fit()
regressor_OLS.summary()


 






