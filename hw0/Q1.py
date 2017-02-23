import numpy as np
import sys

array1 = np.loadtxt(sys.argv[1], delimiter = ',')
array2 = np.loadtxt(sys.argv[2], delimiter = ',')
ans = np.dot(array1,array2)
ans = ans.tolist()
ans.sort()
file = open("ans_one.txt",'w')
for i in range(0, len(ans)):
	file.write(str(int(ans[i])) + '\n')
