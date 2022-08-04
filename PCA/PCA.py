import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data =np.array([[40,20],
                [55,30],
                [70,60],
                [50,55],
                [45,40],
                [62,75],
                [45,30],
                [68,80],
                [80,70],
                [75,90]])
plt.scatter(data[:,0],data[:,1])
plt.xlabel("Maths Marks")
plt.ylabel("Science Marks")
plt.show()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
#print(scaled_data)

pca = PCA(n_components=1)
pca.fit(scaled_data)
pcaVarience = pca.explained_variance_
# print(pcaVarience)
pcaVarienceRatio = pca.explained_variance_ratio_
# print(pcaVarienceRatio)
pca_scaled_data = pca.transform(scaled_data)
# print(scaled_data.shape)
# print(pca_scaled_data.shape)

pca_scaled_data = pca.inverse_transform(pca_scaled_data)

plt.scatter(scaled_data[:,0],scaled_data[:,1],alpha=0.2)
plt.scatter(pca_scaled_data[:,0],pca_scaled_data[:,1])
plt.show()
