import numpy as np
data = np.array([
[26,50000],
[29,70000],
[34,55000],
[31,41000],
])

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaled_data =scaler.fit_transform(data)
#print(scaled_data)

from sklearn.preprocessing import StandardScaler
scaler2 =StandardScaler()
scaled_data2 =scaler2.fit_transform(data)
print(scaled_data2)