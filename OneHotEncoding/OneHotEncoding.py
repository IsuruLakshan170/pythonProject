import pandas as pd
data =pd.read_csv("E:\Python\pythonProject\OneHotEncoding\Book1.csv")
#print(data['results'])

result_category =data['results']
#print(result_category)

from sklearn.preprocessing import LabelEncoder
obj = LabelEncoder()
result =obj.fit_transform(result_category)
#print(result)

from sklearn.preprocessing import LabelBinarizer
obj1 = LabelBinarizer()
result1 =obj1.fit_transform(result_category)
print(result1)
