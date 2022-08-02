import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("E:\Python\pythonProject\svm\iris.csv")
#print(data.head())
#print(data.info())

value_count = data['variety'].value_counts()
#print(value_count)

sns.pairplot(data,hue='variety')
#plt.show()

x = data.drop('variety',axis=1)
#print(x.head())

y = data['variety']
#print(y.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2)

from sklearn.svm import SVC
model = SVC(kernel='poly',C =1 )
model.fit(x_train,y_train)
#print(model.kernel)

pred =model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,pred)
print(accuracy)







