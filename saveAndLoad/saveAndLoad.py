from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dataset = load_iris()
x_train,x_test,y_train,y_test = train_test_split(dataset.data,dataset.target,test_size=0.2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
# print("Accuracy : {} ".format(rf.score(x_test,y_test)))

#use pickle
import pickle

#save model
with open("rf_model1.pickle","wb") as file:
    pickle.dump(rf,file)

#load model
with open("rf_model1.pickle","rb") as file:
    model1 =  pickle.load(file)

#print(model1)

SepalLength = 6.3
SepalWidth = 2.3
PetalLength = 4.6
PetalWidth = 1.3

predValue = model1.predict([[SepalLength,SepalWidth,PetalLength,PetalWidth]])
#print(predValue)

#use joblib
import joblib

#save model
joblib.dump(rf,"rf_model2.pickle")

#load model
model2 = joblib.load("rf_model2.pickle")

SepalLength = 6.3
SepalWidth = 2.3
PetalLength = 4.6
PetalWidth = 1.3

predValue = model2.predict([[SepalLength,SepalWidth,PetalLength,PetalWidth]])
print(predValue)