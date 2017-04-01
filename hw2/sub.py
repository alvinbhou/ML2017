import csv, random, math
from operator import add
import time

model = [-1.70263679169,-1.61653156727,-1.19030263321,-2.25813990584,-1.95049115504,-2.17781907811,-1.89095838624,-0.32957033276,-0.349996922373,0.255085141289,1.27647929533,-0.879405973228,0.603692775108,-9.74161890703,1.09722454908,-0.550980115999,-0.0304951984873,-0.700798326003,-4.42873819162,-0.501079446875,-0.310665601921,-0.970226278978,-0.828714325522,-6.20702979771,-0.970764488172,-1.67445661323,1.19915119033,0.476779768791,-1.65743023123,-2.17884650406,-1.81886625779,-1.70852875295,-0.74485077357,-2.04806373036,-0.728913848665,0.00442601554127,-1.81243639871,-1.48813458088,-1.07245925199,-1.59136364436,-4.41385074721,-0.254176296949,-0.217945852743,-0.512755893327,-0.127863086318,-0.912070815693,-0.986155203531,1.8587363061,2.82757805786,25.9044961188,2.79080428606,0.143887900043,0.556610930413,-0.44999580423,-1.40509904281,-2.9498840132,-0.468413063847,-2.63907993977,-1.24819040204,-1.30973390202,-0.462087879254,-0.127045407397,-0.388548931955,-1.74842792403,-0.992618234502,-1.04877817546,-5.15467352903,-1.78509985825,-0.737977340405,-0.875593145505,-1.15927846715,-0.804936470899,-0.292173869933,0.0204882281724,-0.912080360742,-0.385246288713,-1.23565636604,-1.31736662384,-1.48053232541,-5.83568181025,-1.59438755606,-0.410862058372,-0.824527892388,-0.748770291277,-1.13174504121,-0.525001055849,-1.76039994856,-0.719892924715,-1.25677413092,-1.20937562903,-0.61996143091,-1.89388458146,-0.151782692845,-0.988498961468]
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
		x_vector.append(cData[i].eduStatus + cData[i].workClass + cData[i].marryStatus + cData[i].occupation + cData[i].age + cData[i].hours_per_week + cData[i].capital_gain + cData[i].capital_loss + cData[i].sex + cData[i].country)				
	return (x_vector, y_vector)

test_data = []

test_data = loadData(test_data, 'X_test_norm.csv', 0)
print(test_data[0].marryStatus)
y_pred = [0] * len(test_data)
(x_test_vector, yxx) = selectAttr(test_data)

print(len(x_test_vector[0]))
print(len(model))

for i in range(len(x_test_vector)):
	if(f_wb(x_test_vector[i], model,-0.6677721321874243) >= 0.5):
		y_pred[i] = 1
	else:
		y_pred[i] = 0

with open("result66.csv", "w", newline='') as mFile:
	writer = csv.writer(mFile)
	writer.writerow(["id","label"])
	for i in range(0, len(x_test_vector)):
		mFile.write(str(i+1) + ",")
		mFile.write(str(y_pred[i]))
		mFile.write("\n")