import pandas as pd
data = pd.DataFrame({'Math':[70,60,40,80,30],
                     'Chemistry':[60,80,65,55,60],
                     'Maths':[70,60,40,80,30],
                     'Physics':[50,50,50,50,50],
                     'General Test':[70,70,60,60,80]})
# print(data)

#variance
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0)
selected_features = selector.fit_transform(data)
# print(selected_features)
data = pd.DataFrame(selected_features,columns=selector.get_feature_names_out())
# print(data)

#correlation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cor = data.corr()
# print(cor)

plt.figure(figsize=(8,6))
sns.heatmap(cor,annot = True, cmap ='Wistia')
# plt.show()

corr_features = set()
for i in range(len(cor.columns)):
    for j in range(i):
        if abs(cor.iloc[i,j]>0.9):
            colname = cor.columns[i]
            corr_features.add(colname)

# print(corr_features)

data = data.drop(corr_features, axis=1)
print(data)