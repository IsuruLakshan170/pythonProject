import  numpy as np
from sklearn.model_selection import train_test_split

x = np.array ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array ([0,0,1,0,1,0,0,0,0,0,1,1,1,0,1,0,1,1,1,1])
# print(len(x))
# print(len(y))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
print(x_train)
print(x_test)
print(y_train)
print(y_test)


