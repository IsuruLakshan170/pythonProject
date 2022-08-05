import pandas as pd
import matplotlib as plt
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest

#regression problem
from sklearn.feature_selection import mutual_info_regression

x,y = make_regression(n_samples=50,n_features=5)
x = pd.DataFrame(x)
# print(x.head())

fs = SelectKBest(score_func= mutual_info_regression, k = 3)
fsPrint = fs.fit(x,y)
# print(fsPrint)
fsScorePrint = fs.scores_
# print(fsScorePrint)

mi_score = pd.Series(fs.scores_,index=x.columns)
# print(mi_score)

mi_score.sort_values(ascending=False).plot.bar(figsize=(6,4))
x_selected = fs.fit_transform(x,y)
x_selected = pd.DataFrame(x_selected)
print(x.head())
print(x_selected.head())

#classification problem
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif

x,y = make_classification(n_samples=50,n_features=5,n_informative=2)
x = pd.DataFrame(x)
# print(x.head())
# print(y[:5])


fs = SelectKBest(score_func= mutual_info_classif, k = 3)
fsPrint = fs.fit(x,y)
mi_score = pd.Series(fs.scores_,index=x.columns)
mi_score.sort_values(ascending=False).plot.bar(figsize=(6,4))

x_selected = fs.fit_transform(x,y)
x_selected = pd.DataFrame(x_selected)
print(x.head())
print(x_selected.head())