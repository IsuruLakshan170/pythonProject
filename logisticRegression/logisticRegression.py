import matplotlib.pyplot as plt
import numpy as np

age =np.array([16,24,31,29,40,33,18,15,19,21,29,31,24,20,34,22,32,36,37,75,70])
job = np.array([0,0,1,0,1,0,0,0,0,0,1,1,1,0,1,0,1,1,1,1,1])

# print(len(age))
# print(len(job))

plt.scatter(age,job)
plt.xlabel('Age')
# plt.show()

m,c = np.polyfit(age,job,1)
plt.xlabel('Age')
plt.plot(age,job,'o')
plt.plot(age,m * age + c)
plt.show()

