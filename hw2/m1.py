import csv, random, math
from operator import add
import time
import numpy as np


class Man():
	def __init__(self):
		self.age = []
		self.fnlwgt = []
		self.sex = []
		self.capital_gain = []
		self.capital_loss = []
		self.workClass = []
		self.eduStatus = []
		self.marryStatus = []
		self.occupation = []
		self.relation = []
		self.race = [] 
		self.country = []
		self.id = 0 
		self.flag = 0

def loadData(data, d, flag):
	f = open(d, 'r')
	index = 0
	for row in csv.reader(f):
		if (index == 0):
			attr_header = row
			index = index + 1
			continue
		row = list(map(float, row))		
		m = Man()
		m.age = [row[0]]
		m.fnlwgt = [row[1]]
		m.sex = [row[2]]
		m.capital_gain = [row[3]]
		m.capital_loss = [row[4]]
		m.hours_per_week = [row[5]]
		m.workClass = row[6:15]
		m.eduStatus = row[15:31]
		m.marryStatus = row[31:38]
		m.occupation = row[38:53]
		m.relation = row[53:59]
		m.race = row[59:64]
		m.country = row[64:]
		m.id = index
		data.append(m)
		index = index + 1
	f.close()

	if(flag == 1):
		f = open('Y_train.csv','r')
		index = 0
		for row in csv.reader(f):
			data[index].flag = int(row[0])
			index = index + 1
		f.close()
	
	return data


# gradient descent 
def gradientDescent(iteration,x_vector,y_vector, initPara):	
	minError = 99999
	xLen = len(x_vector[0])
	dataLen = len(x_vector)
	b = 0 # initial b
	w = np.zeros(xLen)
	lr = 10 # learning rate

	b_lr = 0.0
	w_lr = np.zeros(xLen)

	# Store initial values for plotting.
	b_history.append(b)
	w_history.append(w)
	# Iterations
	for it in range(iteration):	    
	    b_grad = 0.0
	    w_grad = np.zeros(xLen)
	    for n in range(dataLen):
	       	f_wbComponent = f_wb(x_vector[n], w, b)
	       	b_grad = b_grad - (y_vector[n] - f_wbComponent)	       	
	       	for i in range(0,xLen): 
	        	w_grad[i] = w_grad[i]  - (y_vector[n] - f_wbComponent)* x_vector[n][i]	 
	    
	    b_lr = b_lr + b_grad**2
	    # Update parameters.
	    b = b - lr/math.sqrt(b_lr) * b_grad 
	    for i in range(0,xLen):
	   		w_lr[i] = w_lr[i] + w_grad[i]**2 + 0.00000001
	   		w[i] = w[i] - lr/math.sqrt(w_lr[i]) * w_grad[i]
	    # Store parameters 
	    b_history.append(b)
	    w_history.append(w)		
	return(b_history[-1], w_history[-1])	

def f_wb(x_n, w, b):
	z = np.dot(x_n, w) + b
	ans = 1/(1.0 +np.exp(-1 * z))
	return np.clip(ans, 0.000000000001, 0.9999999999999)

def classification(data):
	cData = [[],[]]
	for d in data:
		if d.flag == 0:
			cData[0].append(d)
		else:
			cData[1].append(d)
	return cData

def selectAttr(cData):
	# select the attr. we want to take into consideration

	x_vector = []
	y_vector = []
	for i in range(len(cData)):		
		x = cData[i].eduStatus + cData[i].workClass + cData[i].marryStatus + cData[i].occupation  + cData[i].age  + cData[i].hours_per_week + cData[i].capital_gain + cData[i].capital_loss + cData[i].country + cData[i].race + cData[i].sex + cData[i].fnlwgt
		y = cData[i].flag
		x = np.array(x)
		y = np.array(y)
		x_vector.append(x)
		y_vector.append(cData[i].flag)
	return (x_vector, y_vector)



def initPara(x_vector):
	mean_vector = [0] * len(x_vector[0])
	for i in range(len(x_vector)):		
		mean_vector = list(map(add, mean_vector, x_vector[i]))
	mean_vector[:] = [(x / len(x_vector) ) for x in mean_vector]
	return mean_vector

def splitData(data, seed ,ratio):
	dataLen = len(data)
	d = data
	random.seed(seed)
	random.shuffle(d)
	train_data = d[:math.floor(dataLen*ratio)]
	valid_data = d[math.floor(dataLen*ratio):]
	return(train_data, valid_data)


start_time = time.time()

data = []
train_data = []
valid_data = []
w_history = []
b_history = []

# load data
data = loadData(data, 'X_train_norm.csv', 1)

		



# split train/ valid set
seed = 77777
ratio = 0.7
# (train_data, valid_data) = splitData(data, seed, ratio)
# for i in range(len(data)):
# 	print(data[i].id)

# (x_vector, y_vector) = selectAttr(data)
(x_valid_vector, y_valid_vector) = selectAttr(data)
print(len(x_valid_vector[0]))
init_vector = [0.0] * len(x_valid_vector[0])

(opt_b , opt_model) = gradientDescent(800,x_valid_vector,y_valid_vector, init_vector)
print(opt_model)
print(opt_b)


# (x_vector, y_vector) = selectAttr(data)
(x_valid_vector, y_valid_vector) = selectAttr(data)

p = 0
for i in range(len(x_valid_vector)):
	if(f_wb(x_valid_vector[i], opt_model, opt_b) >= 0.5):
		guess = 1
	else:
		guess = 0
	if(guess == y_valid_vector[i]):
		p = p + 1
p = p / (len(x_valid_vector))
print(p)



sAttr = "cData[i].eduStatus + cData[i].workClass + cData[i].marryStatus + cData[i].occupation  + cData[i].age  + cData[i].hours_per_week + cData[i].capital_gain + cData[i].capital_loss + cData[i].country + cData[i].race + cData[i].sex + cData[i].fnlwgt"
with open("model.csv", "a", newline='') as mFile:
	mFile.write(sAttr + " ")
	mFile.write(str(p))
	mFile.write('\n')
	mFile.write(str(opt_b) + " ;")

	writer = csv.writer(mFile)
	writer.writerow(opt_model)
	mFile.write('\n')

# # load test data
# test_data = []

# test_data = loadData(test_data, 'X_test.csv', 0)
# (x_test_vector, yzz) = selectAttr(test_data)
# y_pred = [0] * len(x_test_vector)
# for i in range(len(x_test_vector)):
# 	if(f_wb(x_test_vector[i], opt_model, opt_b) >= 0.5):
# 		y_pred[i] = 1
# 	else:
# 		y_pred[i] = 0

# with open("result5.csv", "w", newline='') as mFile:
# 	writer = csv.writer(mFile)
# 	writer.writerow(["id","label"])
# 	for i in range(0, len(x_test_vector)):
# 		mFile.write(str(i+1) + ",")
# 		mFile.write(str(y_pred[i]))
# 		mFile.write("\n")
	
print("--- %s seconds ---" % (time.time() - start_time))

