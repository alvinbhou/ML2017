import csv, random, math
from operator import add
import time
import numpy as np

model = [-1.79693011795,-1.72310070515,-1.24689744928,-2.27875160236,-1.98913677465,-2.36470348609,-2.01234068595,-0.517781574415,-0.458770589379,0.0856955708013,1.09855898357,-1.00797818866,0.480323946023,-9.22989859034,0.933302369623,-0.683829917257,0.133074311911,-0.56994091652,-4.83004623289,-0.379143876688,-0.229334197067,-0.881455188079,-0.692233820103,-5.97752809305,-1.13338112079,-1.25387798809,0.992103208524,0.375686501457,-1.45450915417,-1.69129260061,-1.37400849614,-1.13460011721,-0.56260208689,-1.46152774076,-0.513539943907,0.218186758087,-1.57264702025,-1.26096247718,-0.874590758338,-1.39626344622,-3.52572068821,-0.0763411658,0.0194549376428,-0.283019054065,0.0717575726389,-0.683104206479,-0.493806667916,1.78953044143,2.73870933848,26.8319348545,2.75347715574,-0.00669864157489,-0.559280851659,-1.65735483492,-2.9787563202,-0.6338555881,-2.78281199859,-1.90056646038,-1.5685495114,-0.655244245322,-0.454219947986,-0.523109589412,-1.95595721278,-1.98472310378,-0.792883211503,-4.22458284943,-2.2216857887,-1.0906998803,-0.823812958738,-1.36981571647,-0.926432720925,-0.122187828395,-0.130072448573,-1.14484756995,-0.668343683571,-1.55985119506,-1.48007773594,-1.34827404236,-6.05174188359,-2.56316183281,-0.605939669612,-0.737641246718,-0.797496790863,-1.20404301317,-0.947223941506,-1.89353005968,-1.22577530931,-1.53143055319,-1.37876413908,-0.767895965759,-2.43523353005,0.421840087243,-1.18696288878,-1.09375847711,-0.338706481455,-0.621320833422,-0.701370099881,-0.460606710081,0.835152688484,-0.107249417147,-0.116210147366,-0.720457816839,-1.26447433563,-0.228365819576,1.17447726774]

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

def f_wb(x_n, w, b):
	z = sum([a*b for a,b in zip(w,x_n)]) + b
	ans = 1/(1+math.exp(-1 * z))
	return ans

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
		elif(capital_gain_list[i][0] <= gain_mean + gain_var * 7):
			flag = 4
		elif(capital_gain_list[i][0] <= gain_mean + gain_var * 9):
			flag = 5
		elif(capital_gain_list[i][0] <= gain_mean + gain_var * 11):
			flag = 6
		elif(capital_gain_list[i][0] <= gain_mean + gain_var * 13):
			flag = 7
		else:
			flag = 8
		cap_g = [0] * 9
		cap_g[flag] = 1
		data[i].capital_gain = cap_g
		# print(data[i].capital_gain)

	capital_loss_list = [o.capital_loss for o in data]
	loss_mean = np.mean(capital_loss_list)
	loss_var = np.var(capital_loss_list)

	print(gain_mean)
	print(gain_var)
	print(loss_mean)
	print(loss_var)
	return data 

def selectAttr(cData):
	# select the attr. we want to take into consideration
	# if(len(cData) == 2):
	# 	x_vector = [[],[]]
	# 	for i in range(2):		
	# 		for j in range(len(cData[i])):
	# 			x_vector[i].append(np.concatenate((cData[i].eduStatus + cData[i].workClass + cData[i].marryStatus + cData[i].occupation),0))
	# else:
	x_vector = []
	y_vector = []
	for i in range(len(cData)):
		y_vector.append(cData[i].flag)
		x_vector.append(cData[i].eduStatus + cData[i].workClass + cData[i].marryStatus + cData[i].occupation + cData[i].age + cData[i].hours_per_week + cData[i].capital_gain + cData[i].capital_loss + cData[i].country + cData[i].race + cData[i].sex + cData[i].relation)				
	return (x_vector, y_vector)



test_data = []

test_data = loadData(test_data, 'X_test_norm.csv', 0)
# test_data = clusterData(test_data)
print(test_data[0].marryStatus)
y_pred = [0] * len(test_data)
(x_test_vector, yxx) = selectAttr(test_data)

print(len(x_test_vector[0]))
print(len(model))

for i in range(len(x_test_vector)):
	if(f_wb(x_test_vector[i], model,-0.959546317195) >= 0.5):
		y_pred[i] = 1
	else:
		y_pred[i] = 0

with open("result67.csv", "w", newline='') as mFile:
	writer = csv.writer(mFile)
	writer.writerow(["id","label"])
	for i in range(0, len(x_test_vector)):
		mFile.write(str(i+1) + ",")
		mFile.write(str(y_pred[i]))
		mFile.write("\n")