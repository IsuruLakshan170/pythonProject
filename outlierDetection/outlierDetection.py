import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data =pd.read_csv("E:\Python\pythonProject\outlierDetection\insurance.csv")
# print(data.head(5))
#print(data.shape)

plt.hist(data['charges'])
plt.xlabel('Charges')
plt.ylabel('Count')
plt.show()
# print(data.describe())

mean = np.mean(data['charges'])
std = np.std(data['charges'])
# print(mean)
# print(std)

zCore = (data['charges']- mean)/std
# print(zCore)

data['charges_z_score'] = (data['charges']-mean)/std
# print(data.head(5))

# print(data[data['charges_z_score']>3])
# print(data[data['charges_z_score']< -3])

# print(data['charges_z_score'].min())
# print(data['charges_z_score'].max())

outlier_indexes = []
outlier_indexes.extend(data.index[data['charges_z_score']>3].tolist())
# print(outlier_indexes)

outlier_indexes.extend(data.index[data['charges_z_score']< -3].tolist())
# print(outlier_indexes)

new_data = data.drop(data.index[outlier_indexes])
# print(new_data)
# print(new_data.shape)

# print(data.shape[0],new_data.shape[0])

new_data = new_data.drop('charges_z_score',axis =1)
#print(new_data.head())

plt.hist(new_data['charges'])
plt.xlabel('Charges')
plt.ylabel('Count')
plt.show()