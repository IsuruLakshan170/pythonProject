import sys

print(sys.version)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("E:\Python\pythonProject\linerRegression_5\Book1.csv")
#print(data)
plt.scatter(data.videos,data.views,color ='red')
plt.xlabel('Number of Videos')
plt.ylabel('Total Views')
#plt.show()

#print(data.views)
#print(data.videos)

x =np.array(data.videos.values)
#print(x)
y=np.array(data.views.values)
#print(y)

model =LinearRegression()
model.fit(x.reshape((-1,1)),y)

new_x = np.array([45]).reshape((-1,1))
#print(new_x)

pred =model.predict(new_x)
#print(pred)

plt.scatter(data.videos,data.views,color ='red')
m,c =np.polyfit(x,y,1)
plt.plot(x,m*x+c)
plt.show()

#print(m)
#print(c)

y_new =m*45+c
print(y_new)
