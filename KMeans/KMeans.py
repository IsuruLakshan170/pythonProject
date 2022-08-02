import numpy as np
import pandas as pd

data = pd.read_csv("E:\Python\pythonProject\KMeans\Mall_Customers.csv")
#print(data.sample(5))

data = data[['Annual Income (k$)','Spending Score (1-100)']]
#print(data.sample(5))

data = data.rename(columns={'Annual Income (k$)':'income','Spending Score (1-100)':'score'})
#print(data.sample(5))

import matplotlib.pyplot as plt
plt.scatter(data['income'],data['score'])
#plt.show()

from sklearn.cluster import KMeans
k_values = [1,2,3,4,5,6,7,8,9,10]
wcss_error =[]
for k in k_values:
    model = KMeans(n_clusters=k)
    model.fit(data[['income','score']])
    wcss_error.append(model.inertia_)

#print(wcss_error)

plt.xlabel('Number of Cluster(k)')
plt.ylabel('WCSS Error')
plt.plot(k_values,wcss_error)
plt.show()

from sklearn.cluster import KMeans
model = KMeans(n_clusters=5)
pred = model.fit_predict(data)
#print(pred)

data['cluster'] = pred
#print(data.head(5))

c1 = data[data['cluster'] == 0]
c2 = data[data['cluster'] == 1]
c3 = data[data['cluster'] == 2]
c4 = data[data['cluster'] == 3]
c5 = data[data['cluster'] == 4]

#print(c1.head(5))

plt.scatter(c1['income'],c1['score'])
plt.scatter(c2['income'],c2['score'])
plt.scatter(c3['income'],c3['score'])
plt.scatter(c4['income'],c4['score'])
plt.scatter(c5['income'],c5['score'])
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color ='black')
plt.show()

KMeanModel = model.cluster_centers_
print(KMeanModel)



