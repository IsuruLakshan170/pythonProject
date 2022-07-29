import numpy as np
import pandas as pd

data = pd.read_csv("E:\Python\pythonProject\KNN_Algorithm\iris.csv")
#print(data.head())
# print(data.shape)
# print(data['variety'].value_counts())
# print(data.info())
# print(data.describe())

x= data.iloc[:,1:4]
#print(x.head())

y= data.iloc[:,-1]
#print(y.head())

from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
x =scaler.fit_transform(x)
#print(x[0:5])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2)
#print(x_train.shape)
#print(x_test.shape)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train,y_train)

pred =model.predict(x_test)
#print(pred[0:5])
#print(y_test[0:5])

from sklearn.metrics import accuracy_score
accuracy =accuracy_score(y_test,pred)
#print(accuracy)

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test,pred)
#print(cm)

result = pd.DataFrame(data=[y_test.values,pred],index = ['y_test','pred'])
#print(result.transpose())

correct_sum=[]
for i in range(1,20):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train,y_train)
    pred =model.predict(x_test)
    correct =np.sum(pred == y_test)
    correct_sum.append(correct)
#print(correct_sum)

result =pd.DataFrame(data=correct_sum)
result.index =result.index +1
result.T
print(result.T)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train,y_train)
pred =model.predict(x_test)
print(accuracy_score(y_test,pred))

z = confusion_matrix(y_test,pred)
print(z)