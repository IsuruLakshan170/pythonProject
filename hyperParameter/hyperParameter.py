import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("E:\Python\pythonProject\svm\iris.csv")
#print(data.head())
#print(data.info())
x = data.drop('variety',axis=1)
#print(x.head())

y = data['variety']
#print(y.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2)

from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)
svcScore =model.score(x_test,y_test)
#print(svcScore)

from sklearn.model_selection import GridSearchCV
param_grid ={
    'C':[0.1,1,10],
    'kernel':['rbf','linear','poly']
}
grid_search = GridSearchCV(estimator=model,param_grid=param_grid)
grid_search.fit(x_train,y_train)
bestParams = grid_search.best_params_
print(bestParams)
grid_searchScore = grid_search.score(x_test,y_test)
print(grid_searchScore)

from sklearn.model_selection import RandomizedSearchCV
param_dist ={
    'C':[0.1,1,10],
    'kernel':['rbf','linear','poly']
}
randomized_search = RandomizedSearchCV(estimator=model,param_distributions=param_dist,n_iter=8)
randomized_search.fit(x_train,y_train)

randBestParams = randomized_search.best_params_
print(randBestParams)
randScore = randomized_search.score(x_test,y_test)
print(randScore)
