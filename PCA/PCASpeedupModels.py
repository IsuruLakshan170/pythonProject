from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
from sklearn.decomposition import PCA

digits = load_digits()
#print(digits.data.shape)

sc = StandardScaler()
new_data = sc.fit_transform(digits.data)
#print(scaled_data)


pca = PCA(n_components = 10)
new_data_pca = pca.fit_transform(new_data)

x_train,x_test,y_train,y_test = train_test_split(new_data_pca,digits.target,test_size=0.2, random_state=42)
model = LogisticRegression(solver='lbfgs',max_iter=1000)

start = time.time()
model.fit(x_train,y_train)
end = time.time()
print((end - start)*1000)

y_pred = model.predict(x_test)
PCAAccuracy = accuracy_score(y_test,y_pred)
print(PCAAccuracy)
