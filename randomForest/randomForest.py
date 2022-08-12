import numpy as np
import pandas as pd

data = pd.read_csv("E:\Python\pythonProject\desionTree\kyphosis.csv")
#print(data.head())
#print(data.shape)
#print(data.info())

x = data.drop('Kyphosis',axis=1)
#print(x.head())

y = data['Kyphosis']
#print(y.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50)
model.fit(x_train,y_train)

pred =model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,pred)
print(accuracy)



