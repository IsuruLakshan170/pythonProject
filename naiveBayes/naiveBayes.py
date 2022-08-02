import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("E:\Python\pythonProject\desionTree\kyphosis.csv")
#print(data.head())
#print(data.info())

ax = sns.pairplot(data,hue="Kyphosis")
ax.set( xlabel='Days', ylabel='Amount_spend' )
plt.title( 'My first graph' )
#plt.show()

x = data.drop('Kyphosis',axis=1)
#print(x.head())

y = data['Kyphosis']
#print(y.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)


pred =model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,pred)
print(accuracy)
