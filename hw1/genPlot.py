import matplotlib.pyplot as plt
import csv
import numpy as np

error = []
num = []
f = open('errorandnum.csv', 'r')
for row in csv.reader(f):
	error.append(row[0])
	num.append(row[1])
f.close()
plt.plot(num, error, 'ro')
for a,b in zip(num, error): 
    plt.text(a, b, str("{:.7f}".format(float(b))))

num = list(map(float, num))

plt.xlabel('Number of training data')
plt.xticks(np.arange(min(num)-136, max(num)+300, 100))
plt.ylabel('MRSE')

error = list(map(float, error))
avg = sum(error)/len(error)
print(avg)
x = np.arange(3500)
plt.plot(x, [avg] * len(x), '-')
plt.show()
