import numpy as np
import time


start_time = time.time()
x = [1] * 200
y = [2] * 200
x = np.array(x)
y = np.array(y)
for i in range(1000000):
	# z = sum([a*b for a,b in zip(x,y)])
	z = np.dot(x,y)
	print(z)

print("--- %s seconds ---" % (time.time() - start_time))