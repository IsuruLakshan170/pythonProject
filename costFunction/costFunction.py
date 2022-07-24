import numpy as np
import matplotlib.pyplot as plt

n=5
x =np.array([1,2,3,4,5])
y =np.array([5,8,11,14,17])

m =0
c =0

learning_rate =0.01

for i in range(1,101):
    y_predict = m * x + c
    cost = (1/n) * sum([value ** 2 for value in (y -y_predict)])

    plt.scatter(m, cost)

    #calculate the gradients
    dm = -(2/n) * sum (x * (y -y_predict))
    dc = -(2/n) * sum (y -y_predict)

    # update the parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    print("m {} ,c {} cost{} iteration {}" .format(m,c,cost,i))
plt.show()