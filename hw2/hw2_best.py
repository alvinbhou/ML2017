import csv, random, math, copy
import time, sys
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
		m.age = row[0]
		m.fnlwgt = [row[1]]
		m.sex = [row[2]]
		m.capital_gain = row[3]
		m.capital_loss = row[4]
		m.hours_per_week = row[5]
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
		f = open(Y_trainFile,'r')
		index = 0
		for row in csv.reader(f):
			data[index].flag = int(row[0])
			index = index + 1
		f.close()
	
	return data

def normalData(data):
	capital_gain_list = [o.capital_gain for o in data]
	cg_max = max(capital_gain_list)
	cg_min = min(capital_gain_list)
	capital_loss_list = [o.capital_loss for o in data]
	cl_max = max(capital_loss_list)
	cl_min = min(capital_loss_list)
	hours_week_list = [o.hours_per_week for o in data]
	hw_max = max(hours_week_list)
	hw_min = min(hours_week_list)
	age_list = [o.age for o in data]
	age_max = max(age_list)
	age_min = min(age_list)

	for i in range(len(data)):
		data[i].capital_gain = [(capital_gain_list[i] - cg_min) / (cg_max - cg_min)]
		data[i].capital_loss = [(capital_loss_list[i] - cl_min) / (cl_max - cl_min)]
		data[i].hours_per_week = [(hours_week_list[i] - hw_min) / (hw_max - hw_min)]
		data[i].age = [(age_list[i] - age_min) / (age_max - age_min)]
	
	return data	

def clusterData(data):
	dataLen = len(data)
	capital_gain_list = [o.capital_gain for o in data]
	gain_mean = np.mean(capital_gain_list)
	gain_var = np.var(capital_gain_list)
	for i in range(dataLen):
		flag = 0
		if(capital_gain_list[i][0] <= 0.00000001):
			flag = 0
		elif(capital_gain_list[i][0] <= gain_mean + gain_var):
			flag = 1
		elif(capital_gain_list[i][0] <= gain_mean + gain_var * 3):
			flag = 2
		elif(capital_gain_list[i][0] <= gain_mean + gain_var * 5):
			flag = 3
		elif(capital_gain_list[i][0] <= 0.6):
			flag = 4
		elif(capital_gain_list[i][0] <= 0.6):
			flag = 5
		elif(capital_gain_list[i][0] <= 0.9):					
			flag = 7
		else:
			flag = 8
		flag = 0		
		cap_g = [0] * 9
		cap_g[flag] = 1
		data[i].capital_gain = cap_g
		# print(data[i].capital_gain)

	capital_loss_list = [o.capital_loss for o in data]
	loss_mean = np.mean(capital_loss_list)
	loss_var = np.var(capital_loss_list)	
	return data 


# gradient descent 
def gradientDescent(iteration,x_vector,y_vector, initPara, data):	
	maxAccu = 0.0
	# vector length
	xLen = len(x_vector[0])
	# training data length
	dataLen = len(x_vector)
	b = 0 # initial b
	w = initPara # w parameters
	lr = 1 # learning rate

	b_lr = 0.0
	w_lr = np.zeros(xLen)

	opt_model = None
	opt_b = None

	lamb = 0.5

	# Store initial values for plotting.
	b_history.append(b)
	w_history.append(w)
	# Iterations
	for it in range(iteration):	    
	    b_grad = 0.0
	    w_grad = np.zeros(xLen)
	    # print("%s  gradient--- %s seconds ---" % (it, (time.time() - start_time)))
	    for n in range(dataLen):
	       	f_wbComponent = f_wb(x_vector[n], w, b)
	       	b_grad = b_grad - (y_vector[n] - f_wbComponent)	     
	       	w_grad = w_grad - (y_vector[n] - f_wbComponent) * x_vector[n]	    
	       	# for i in range(0,xLen): 
	        # 	w_grad[i] = w_grad[i]  - (y_vector[n] - f_wbComponent)* x_vector[n][i]
	    b_lr = b_lr + b_grad**2
	    # Update parameters 
	    b = b - np.multiply(lr/np.sqrt(b_lr),b_grad)
	    w_lr = w_lr + np.square(w_grad) + 0.00000000000001
	    w = w - np.multiply(lr/ np.sqrt(w_lr),w_grad)
	    # Store parameters 
	    b_history.append(b)
	    w_history.append(w)
	    if(it % 10 == 0):
		    p = errorChecking(data, w, b, it)
		    if( p > maxAccu):
		    	opt_b = b
		    	opt_model = w
		    	maxAccu = p

	    # print("--- %s seconds ---" % (time.time() - start_time))
	return(opt_b, opt_model, maxAccu)	

# sigmoid function
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
		x = cData[i].eduStatus + cData[i].workClass + cData[i].marryStatus + cData[i].occupation + cData[i].age + cData[i].hours_per_week + cData[i].capital_gain + cData[i].capital_loss + cData[i].country + cData[i].race + cData[i].sex + cData[i].relation
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
	d = copy.copy(data)
	random.seed(seed)
	random.shuffle(d)
	train_data = d[:math.floor(dataLen*ratio)]
	valid_data = d[math.floor(dataLen*ratio):]
	return(train_data, valid_data)

def errorChecking(data, opt_model, opt_b, it):
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
	print("Iteration %s , accuracy %s" % (it, p))
	return p


rawDataFile = sys.argv[1]
testDataFile = sys.argv[2]
X_trainFile = sys.argv[3]
Y_trainFile = sys.argv[4]
X_testFile = sys.argv[5]
predictionFile = sys.argv[6]


# start_time = time.time()

data = []
train_data = []
valid_data = []
w_history = []
b_history = []

# load data
data = loadData(data, X_trainFile, 1)
data = normalData(data)
# data = clusterData(data)	

# split train/ valid set
seed = 91322
ratio = 0.9
(train_data, valid_data) = splitData(data, seed, ratio)

(x_valid_vector, y_valid_vector) = selectAttr(train_data)
# print(len(x_valid_vector[0]))
init_vector = np.zeros(len(x_valid_vector[0]))
	

# print("Start gradient--- %s seconds ---" % (time.time() - start_time))
(opt_b , opt_model, p) = gradientDescent(1000,x_valid_vector,y_valid_vector, init_vector, data)
print(opt_model)
print(opt_b)
print(p)



# sAttr = "cData[i].eduStatus + cData[i].workClass + cData[i].marryStatus + cData[i].occupation + cData[i].age + cData[i].hours_per_week + cData[i].capital_gain + cData[i].capital_loss + cData[i].country + cData[i].race + cData[i].sex + cData[i].relation"
# with open("model5.csv", "a", newline='') as mFile:
# 	mFile.write(sAttr + " " + str(seed) + " " + str(seed2) + " " + str(ratio) + " ")
# 	mFile.write(str(p))
# 	mFile.write('\n')
# 	mFile.write(str(opt_b) + " ;")
# 	writer = csv.writer(mFile)
# 	writer.writerow(opt_model)		
		
	
# print("--- %s seconds ---" % (time.time() - start_time))

test_data = []
test_data = loadData(test_data, X_testFile, 0)
test_data = normalData(test_data)
y_pred = [0] * len(test_data)
(x_test_vector, yxx) = selectAttr(test_data)

for i in range(len(x_test_vector)):
	if(f_wb(x_test_vector[i], opt_model,opt_b) >= 0.5):
		y_pred[i] = 1
	else:
		y_pred[i] = 0

with open(predictionFile, "w", newline='') as mFile:
	writer = csv.writer(mFile)
	writer.writerow(["id","label"])
	for i in range(0, len(x_test_vector)):
		mFile.write(str(i+1) + ",")
		mFile.write(str(y_pred[i]))
		mFile.write("\n")