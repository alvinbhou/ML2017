import numpy as np

array1 = np.loadtxt("matrixA.txt", delimiter = ',')
array2 = np.loadtxt("matrixB.txt", delimiter = ',')
ans = np.dot(array1,array2)
ans = ans.tolist()
ans.sort()
print(ans)