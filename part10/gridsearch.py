# improving model performance 
# finding the optimal value for the machine learning 
# the first type of parameter that are learned through the machine learning algorithm
# and the second type of parameter we choose

# grid search will help  you to choose : linear vs nonlinear


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\"

dataset = pd.read_csv(path+"Social_Network_Ads.csv")

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values


# Splitting trainng and test set
from sklearn.model_selection import train_test_split
X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25 , random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting kernel SVM to the Training set

from sklearn.svm import SVC
classifier =SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

# predicting the test set result 
y_pred = classifier.predict(X_test)


# making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)



# applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuries= cross_val_score(estimator = classifier , X= X_train, y = y_train ,cv =10)
accuries.mean()
accuries.std()



# applying Grid search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV

parameters = [
        {
            'C':[1,10,100,1000],
            'kernel': ['linear']
        },
        {
            'C':[1,10,100,1000],
            'kernel': ['rbf'],
            'gamma' : [0.5,0.1,0.01,0.001],
        },
        {
            'C':[1,10,100,1000],
            'kernel': ['rbf'],
            'gamma' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        },
        ]

gridsearch = GridSearchCV(estimator=classifier, 
                          param_grid=parameters,
                          scoring = 'accuracy',
                          cv=10,
                          n_jobs= -1)

gridsearch = gridsearch.fit(X_train,y_train)


best_accuracy = gridsearch.best_score_
best_parameters = gridsearch.best_params_



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



