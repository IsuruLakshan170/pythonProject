import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("E:\Python\pythonProject\linerRegression_5\Book2.csv")
#print(data)
model = LinearRegression()
model.fit(data[['videos','days','subcribers']],data.views)

y_new = model.predict([[45,180,3100]])
print(y_new)

#print(model.coef_)
#print(model.intercept_)

y_new2 =model.coef_[0]*45+model.coef_[1]*180 +model.coef_[2]*3100+model.intercept_

print(y_new2)
