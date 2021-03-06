# k flod cross validation 
# evaluating ourt model performance and improving 
# first the parameter that the model used for learning
# the optimal parameters
# PROBLEM * varience problem 

# good accuracy and low variance = low bias low variance
# large accuracy and high variance = low bias high variance 
# low accuracy and low variance = low bias low variance
# low accuracy and high variance = high bias high variance 


#classification template
# with logistic regression 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
path = "C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\"

dataset = pd.read_csv(path+"Social_Network_Ads.csv")
X = dataset.iloc[:,2:-1].values
Y = dataset.iloc[:,-1].values

# split data set in training set and test set

from sklearn.model_selection import train_test_split
X_train ,X_test , y_train, y_test = train_test_split(X,Y,test_size= 0.25, random_state=0)
 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

 # penalty parametes are used  
# Fitting logistic rgression to the training set
# rcreate your  calssifire right here
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 0)
classifier.fit(X_train,y_train)






#prediction the test set resutls
y_pred = classifier.predict(X_test)


# making the cnfusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


print(cm)



# applying k-fold cross valiedation 
from sklearn.model_selection import cross_val_score

# return accuracy for 10 combination    
# this will get 10 accuracies   
# n_jobs : if you are working with a very hight amount of data 
# n_jobs = -1 will use all the cpy of your pc to calculate the result 
accuracies = cross_val_score(estimator= classifier,X = X_train, y= y_train,cv = 10,n_jobs = -1)

avg_accuracies = accuracies.mean()
std_accuracies = accuracies.std()





from matplotlib.colors import ListedColormap

X_set,y_set = X_train,y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[: , 0].min() - 1 ,stop = X_set[:, 0].max() + 1, step=0.01),
                    np.arange(start = X_set[: , 1].min() - 1 ,stop = X_set[:, 1].max() + 1, step=0.01))

plt.contour(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha= 0.75,cmap = ListedColormap (('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())



for i,j in enumerate(np.unique(y_set)):
    
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
                c= ListedColormap(('red','green'))(i),label=j)
    
    
    
plt.title('logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# visualising the test set results


from matplotlib.colors import ListedColormap

X_set,y_set = X_test,y_test
X1,X2 = np.meshgrid(np.arange(start = X_set[: , 0].min() - 1 ,stop = X_set[:, 0].max() + 1, step=0.01),
                    np.arange(start = X_set[: , 1].min() - 1 ,stop = X_set[:, 1].max() + 1, step=0.01))

plt.contour(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha= 0.75,cmap = ListedColormap (('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())



for i,j in enumerate(np.unique(y_set)):
    
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
                c= ListedColormap(('red','green'))(i),label=j)
    
    
    
plt.title('logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()







