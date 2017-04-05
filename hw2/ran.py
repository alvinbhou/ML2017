import random, copy

data = [1,2,3,4,5,6,7,8,9,0]
for i in range(0,10):
	d = copy.copy(data)
	random.seed(i)
	random.shuffle(d)
	print(d)
	print(data)
