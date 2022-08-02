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

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knnScore = cross_val_score(knn,x,y,cv=5).mean()
print(knnScore)

from sklearn.ensemble import RandomForestClassifier
ranf = RandomForestClassifier()
ranfScore = cross_val_score(ranf,x,y,cv=5).mean()
print(ranfScore)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nbScore = cross_val_score(nb,x,y,cv=5).mean()
print(nbScore)

from sklearn.svm import SVC
svm = SVC()
svmScore = cross_val_score(nb,x,y,cv=5).mean()
print(svmScore)



