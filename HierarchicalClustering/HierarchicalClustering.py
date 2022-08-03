import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sc

data = pd.DataFrame(data = {'x':[0,1.1,1,2,2,4,5,5],
                            'y':[0,1.5,4,2,3,1,0,4]})
#print(data)

plt.scatter(data.x,data.y)
plt.show()

numbers = [0,1,2,3,4,5,6,7]
for indx,value in enumerate(numbers):
    plt.annotate(value,(data.x[indx],data.y[indx]),size=12)

dendrogram =sc.dendrogram((sc.linkage(data,method='ward')))
plt.title('Dendrogram')
plt.show()

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
pred = model.fit_predict(data)
#print(pred)

#print(data)
data['cluster'] = pred
#print(data)

cluster1 = data[data['cluster']==0]
cluster2 = data[data['cluster']==1]
cluster3 = data[data['cluster']==2]

plt.scatter(cluster1.x,cluster1.y)
plt.scatter(cluster2.x,cluster2.y)
plt.scatter(cluster3.x,cluster3.y)
plt.show()




