import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("E:\Python\pythonProject\logisticRegression\Book2.csv")
#print(data)
plt.scatter(data.age,data.job)
#plt.show()

x =data[["age"]]
y =data["job"]
#print(x)
#print(y)

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2)
model =LogisticRegression()
model.fit(x_train,y_train)

model.score(x_test,y_test)
#print(model.score(x_test,y_test))

ages =np.array([[24],[35],[29]])
model.predict(ages)
print(model.predict(ages))