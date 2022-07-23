# # one dimensional example
# from numpy import array
# # list of data
# data = [11, 22, 33, 44, 55]
# # array of data
# data = array(data)
# print(data)
# print(type(data))

# # two dimensional example
# from numpy import array
# # list of data
# data = [[11, 22],
# 		[33, 44],
# 		[55, 66]]
# # array of data
# data = array(data)
# print(data)
# print(type(data))

# # simple indexing
# from numpy import array
# # define array
# data = array([11, 22, 33, 44, 55])
# # index data
# print(data[-1])
# print(data[-5])

# # 2d indexing
# from numpy import array
# # define array
# data = array([[11, 22], [33, 44], [55, 66]])
# # index data
# print(data[0,0])

# # simple slicing
# from numpy import array
# # define array
# data = array([11, 22, 33, 44, 55])
# print(data[2:])
#
# # split input and output
# from numpy import array
# # define array
# data = array([[11, 22, 33],
# 		[44, 55, 66],
# 		[77, 88, 99]])
# # separate data
# X, y = data[:, :-1], data[:, -1]
# print(X)
# print(y)

# # split train and test
# from numpy import array
# # define array
# data = array([[11, 22, 33],
# 		[44, 55, 66],
# 		[77, 88, 99]])
# # separate data
# split = 2
# train,test = data[:split,:],data[split:,:]
# print(train)
# print(test)

# array shape
from numpy import array
# list of data
data = [[11, 22],
		[33, 44],
		[55, 66]]
# array of data
data = array(data)
print(data.shape)