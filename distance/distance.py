from scipy.spatial import distance

A = (5,3)
B = (2,4)

# euclidean distance
d = distance.euclidean(A,B)
print(d)

# manhattan distance
mD = distance.cityblock(A,B)
print(mD)

# cosine distance
cD =distance.cosine(A,B)
print(cD)