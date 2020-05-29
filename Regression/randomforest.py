
# building decision tree
# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
 
# Feature Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Decision tree Regression Model to the dataset
# Create your regressor here

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=500,random_state=0)
regressor.fit(X,y)



# n_estimators = the number of trees






# Predicting a new result
y_pred = regressor.predict([[6.5]])
 


# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



xgrid = np.rrage( min (x), max ( x) , 0.001)
xgrid = xgrid.reshape((len(xgrid),1))
plt.scatter(X,y,color='red')

plt.plot(Xgridd,regressor.predictt(x ))


"""
if you wnat to deliver a mode to the production level then this is the model you want to submit
D.D.spend           if the sign is positive proportional relation
marketing spend     if the sign is negative then inverse proportional relation

magnetude is alway ticky in regression becareful

example:
    like marketing spend's magnitude is greater then RD spend so definitely RD spends has bigger impacts
    but I can easily 
    instead of doller we use sents
    0.79 for every unit if you keep all other variable constant
    if you increase R and D Spend 1 doller you can increase the prediction 
    resutl 0.79 sents
    
    
    forecast things and challagens
    
    
    
    
"""






