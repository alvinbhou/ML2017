import csv, random, math, sys
from operator import add
import numpy as np
from numpy.linalg import inv


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
		self.flag = 0

def loadData(data, d, flag):
	f = open(d, 'r')
	index = 0
	for row in csv.reader(f):
		if (index == 0):
			attr_header = row
			index = index + 1
			continue
		row = list(map(int, row))
		row = np.array(row)
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

def splitData(data):
	dataLen = len(data)
	random.seed(7122)
	random.shuffle(data)
	train_data = data[:math.floor(dataLen*0.7)]
	valid_data = data[math.floor(dataLen*0.7):]
	return(train_data, valid_data)

	
def gaussianDistribution(x,mean,sigma):
	pi = math.pi
	w = np.array([x - mean])
	print(w)
	print(w.transpose())
	print(np.matrix(sigma))
	print(inv(np.matrix(sigma)))
	para1 = 1 / ((2 * pi) ** (1/2))
	para2 = 1 / ((np.linalg.det(sigma)) ** (1/2))
	r = np.matmul(w,inv(sigma))
	para3 = math.exp(np.matmul(r, w.transpose()))
	ans = para1 * para2 * para3
	print(ans)
	return ans

def maxLikelihood(x_vector, flag):
	# mean_vector = [0] * len(x_vector[0])

	# for i in range(len(x_vector)):		
	# 	mean_vector = list(map(add, mean_vector, x_vector[i]))
	# mean_vector[:] = [(x / len(x_vector) ) for x in mean_vector]

	# compute mean vector
	length = len(x_vector)
	mean_vector = np.zeros(len(x_vector[0]))
	for i in range(length):
		mean_vector = mean_vector + x_vector[i]
	mean_vector = mean_vector / length

	# compute covariance vector
	sigma = np.zeros((len(x_vector[0]),len(x_vector[0])))
	# for i in range(length):
	# 	x = np.array([(x_vector[i] - mean_vector)])
	# 	sigma = sigma + np.dot(x.transpose(), x)
	# sigma = sigma/length

	
	# flag 0 for class 0, 1 for class 1
	return ((mean_vector, sigma, flag))

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
	if(len(cData) == 2):
		x_vector = [[],[]]
		for i in range(2):		
			for j in range(len(cData[i])):
				x_vector[i].append(np.concatenate((cData[i][j].workClass,cData[i][j].marryStatus, cData[i][j].race, cData[i][j].occupation, cData[i][j].country, cData[i][j].eduStatus),0))
	else:
		x_vector = []
		for i in range(len(cData)):
				x_vector.append(np.concatenate((cData[i].workClass,cData[i].marryStatus, cData[i].race, cData[i].occupation, cData[i].country, cData[i].eduStatus),0))
	return x_vector

def bernoulli(x,mean):
	# At C1
	# workclass 0:9
	p1 = 0
	for i in range(0,9):
		p1 = p1 + x[i] * mean[i]
	
	# marry 9:16
	p2 = 0
	for i in range(9,16):
		p2 = p2 + x[i] * mean[i]
	
	# # race 16:21
	p3 = 0
	for i in range(16,21):
		p3 = p3 + x[i] * mean[i]
	
	# occupation 21:36
	p4 = 0
	for i in range(21,36):
		p4 = p4 + x[i] * mean[i]
	
	# country 42 elements
	p5 = 0
	for i in range(36,78):
		p5 = p5 + x[i] * mean[i]
	
	p6 = 0 
	# education status 
	for i in range(78,94):
		p6 = p6 + x[i] * mean[i]
	
	
	return p1 * p2  * p3 *  p4 * p5 * p6

def prob(x, x_vector, mean0, mean1, feature):
	P = 0
	P_C0 = len(x_vector[0]) / (len(x_vector[0]) + len(x_vector[1]))
	P_C1 = len(x_vector[1]) / (len(x_vector[0]) + len(x_vector[1]))
	if(feature == 0):
		P = bernoulli(x,mean0) * P_C0 / (bernoulli(x,mean0) * P_C0 + bernoulli(x,mean1) * P_C1)		
	else:
		P = bernoulli(x,mean1) * P_C1 / (bernoulli(x,mean0) * P_C0 + bernoulli(x,mean1) * P_C1)	
	return np.clip(P, 0.0000000001, 0.999999999999)


rawDataFile = sys.argv[1]
testDataFile = sys.argv[2]
X_trainFile = sys.argv[3]
Y_trainFile = sys.argv[4]
X_testFile = sys.argv[5]
predictionFile = sys.argv[6]



data = []
train_data = []
valid_data = []
# load data
data = loadData(data, X_trainFile, 1)

# split train/ valid set
# (train_data, valid_data) = splitData(data)
# classificate train set
cData = classification(data)
# select the desired attr.
x_main_vector = selectAttr(cData)
# print(len(x_vector[0]))


# for testing
# x = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]),np.array([0,0,1]),np.array([0,0,1]),np.array([0,0,1])]

# compute opt. mean, cov matrix of class 0
(mean_vector_0, sigma, flag) = maxLikelihood(x_main_vector[0],0)
(mean_vector_1, sigma, flag) = maxLikelihood(x_main_vector[1],1)

# select the desired attr.
x_vector_valid = selectAttr(data)


# # load test data
test_data = []
test_data = loadData(test_data, X_testFile, 0)

# select the desired attr.
x_test_vector = selectAttr(test_data)


y_pred = []
for i in range(len(x_test_vector)):
	if(prob(x_test_vector[i], x_main_vector, mean_vector_0, mean_vector_1,0) > 0.45):
		y_pred.append(0)		
	else:
		y_pred.append(1)

with open(predictionFile, "w", newline='') as mFile:
	writer = csv.writer(mFile)
	writer.writerow(["id","label"])
	for i in range(0, len(x_test_vector)):
		mFile.write(str(i+1) + ",")
		mFile.write(str(y_pred[i]))
		mFile.write("\n")

# #test accuracy
# p = 0
# for i in range(len(x_test_vector)):
# 	if(y_pred[i] == data[i].flag):
# 		p = p + 1
# print(p)
# print(len(x_test_vector))
# print( p / len(x_test_vector))



