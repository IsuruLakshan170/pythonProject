import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
# print(digits.data)
# print(digits.data.shape)

plt.matshow(digits.images[1])
plt.show()
targetData = digits.target[1]
#print(targetData)

pca = PCA(n_components=3)
new_digits = pca.fit_transform(digits.data)
# print(new_digits.shape)
# print(digits.data.shape)

plt.scatter(new_digits[:,0],new_digits[:,1], c =digits.target)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.colorbar()
plt.show()