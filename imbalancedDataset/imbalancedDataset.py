import numpy as np
import pandas as pd

data = pd.read_csv("E:\Python\pythonProject\imbalancedDataset\kyphosis.csv")
# print(data.head())

x = data.drop('Kyphosis',axis=1)
y = data['Kyphosis']
# print(x)
# print(y)

# print(y.value_counts())

#undersampling

from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler()

x_under , y_under = undersample.fit_resample(x,y)
# print(y_under.value_counts())

#oversampling

from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler()

x_over , y_over = oversample.fit_resample(x,y)
# print(y_over.value_counts())

#smote

from imblearn.over_sampling import SMOTE
smote = SMOTE()

x_smote , y_smote = smote.fit_resample(x,y)
print(y_smote.value_counts())
