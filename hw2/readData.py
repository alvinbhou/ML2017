import csv, random, math
from operator import add
import numpy as np
from numpy.linalg import inv

class man():
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

def loadData(data):
	f = open('X_train.csv', 'r')
	index = 0
	for row in csv.reader(f):
		if (index == 0):
			attr_header = row
			index = index + 1
			continue
		row = list(map(int, row))
		row = np.array(row)
		m = man()
		m.age = [row[0]]
		m.fnlwgt = [row[1]]
		m.sex = [row[2]]
		m.capital_gain = [row[3]]
		m.capital_loss = [row[4]]
		m.hours_per_week = [row[5]]
		m.eduStatus = row[6:15]
		m.workClass = row[15:31]
		m.marryStatus = row[31:38]
		m.occupation = row[38:53]
		m.relation = row[53:59]
		m.race = row[59:64]
		m.country = row[64:]
		data.append(m)
		index = index + 1
	f.close()
	f = open('Y_train.csv','r')
	index = 0
	for row in csv.reader(f):
		data[index].flag = int(row[0])
		index = index + 1
	f.close()
	
	return data

def splitData(data):
	dataLen = len(data)
	random.seed(50)
	random.shuffle(data)
	train_data = data[:math.floor(dataLen*0.7)]
	valid_data = data[math.floor(dataLen*0.7):]
	return(train_data, valid_data)

	
def gaussianDistribution(x,mean,sigma):
	# print(x)
	# print(mean)
	pi = math.pi
	
	para1 = 1 / ((2 * pi) ** (1/2))
	para2 = 1 / ((np.linalg.det(sigma)) ** (1/2))
	para3 = math.exp((-1/2)*  ( np.dot(np.dot((x - mean),inv(sigma) ),  (x-mean).transpose())))
	ans = para1*para2*para3
	# print(ans)
	return ans

def maxLikelihood(x_vector, flag):
	# mean_vector = [0] * len(x_vector[0])

	# for i in range(len(x_vector)):		
	# 	mean_vector = list(map(add, mean_vector, x_vector[i]))
	# mean_vector[:] = [(x / len(x_vector) ) for x in mean_vector]

	length = len(x_vector)
	mean_vector = np.array([0] * len(x_vector[0]))
	for i in range(length):
		mean_vector = mean_vector + x_vector[i]
	mean_vector = mean_vector / length

	sigma = np.zeros((len(x_vector[0]),len(x_vector[0])))
	for i in range(length):
		x = np.array([(x_vector[i] - mean_vector)])
		print(x)
		sigma = sigma + np.dot(x.transpose(), x)
	sigma = sigma/length

	

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
	x_vector = [[],[]]
	for i in range(2):		
		for j in range(len(cData[i])):
			x_vector[i].append(np.concatenate((cData[i][j].eduStatus,cData[i][j].marryStatus),0))

	return x_vector

attr_header = []
data = []
train_data = []
valid_data = []


data = loadData(data)
(train_data, valid_data) = splitData(data)
cData = classification(train_data)
x_vector = selectAttr(cData)
print(len(x_vector[0]))

# for testing
x = [np.array([1,2,3,6]),np.array([5,6,8,8]),np.array([4,2,1,6])]


(mean_vector, sigma, flag) = maxLikelihood(x,0)


# following went wrong
for i in range(len(x_vector[0])):
	gaussianDistribution(x_vector[0][i], mean_vector, sigma)


# z = gaussianDistribution(np.array([80,70]),np.array([75, 71]),np.array([[874,327],[327,929]]))
# print(z)