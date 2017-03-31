import csv, random, math
from operator import add
import time

model = [-1.6088496727830721,-1.541155112642145,-1.1057176916750466,-2.1632065845172597,-1.8609494266495183,-2.095230006126708,-1.7957492936293054,-0.24521607225114378,-0.26766261675284775,0.33997622257442023,1.372518147677843,-0.793567297449567,0.6855452197990481,-12.270391815537897,1.18460899089318,-0.461227316631064,0.0908236158545571,-0.598187333773533,-4.899972381014815,-0.40325025690246297,-0.22064481873974348,-0.8776958846737789,-0.7257000667519868,-6.808094368850253,-0.8785338213878366,-1.6208222520491253,1.2607614985478617,0.5446548849899325,-1.5925611915423532,-2.1328767546576564,-1.7484609613148376,-1.6527833388989792,-0.6609457438190753,-1.8547415350589613,-0.6487605772460965,0.08981861575041031,-1.745382381268321,-1.4005572762317273,-0.9854762950094985,-1.4941921150027886,-4.838639482606251,-0.17051625776434862,-0.12284695031752675,-0.4303818113310265,-0.046734398947950534,-0.8207608994025183,-0.8959458531578055,1.8459651141232445,2.8694457351285583,29.98192711390269,2.8365762547636617,0.6168124436774903,-0.3578935805028152,-1.3694882606112,-2.8463592764827075,-0.37176317989665575,-2.43159724492761,-1.0333231287020779,-1.2587760152990595,-0.37645646764358376,-0.054739445715681054,-0.3029188665674062,-1.705794765136514,-0.9042607965339186,-0.8122771521137988,-5.712087438864613,-1.692478335473457,-0.6701903735298327,-0.8029101142942069,-1.0865482809686224,-0.691877022115834,-0.21109876787223913,0.10192433130029796,-0.689502530513102,-0.30937338283031557,-1.1884465796588646,-1.2011871662085294,-1.3452676149452512,-6.4026069068629745,-1.4915823707409364,-0.35396338632382973,-0.7447781698906126,-0.6644594518731008,-0.9564979675068966,-0.4560433940088128,-1.7051884793337655,-0.6429203243633148,-1.1956574692288304,-1.0128187657134644,-0.5226748744064894,-1.8807961464067124,-0.06326763110738655,-0.882933228463128,-1.1421739716556838,-0.5044488899928521,-0.6830079720179648,-0.9649920233747619,-0.5429182861653814,0.13468313848160762]
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
		x_vector.append(cData[i].eduStatus + cData[i].workClass + cData[i].marryStatus + cData[i].occupation  + cData[i].age  + cData[i].hours_per_week + cData[i].capital_gain + cData[i].capital_loss + cData[i].country + cData[i].race + cData[i].sex)				
	return (x_vector, y_vector)

test_data = []

test_data = loadData(test_data, 'X_test_norm.csv', 0)
print(test_data[0].marryStatus)
y_pred = [0] * len(test_data)
(x_test_vector, yxx) = selectAttr(test_data)

print(len(x_test_vector[0]))
print(len(model))

for i in range(len(x_test_vector)):
	if(f_wb(x_test_vector[i], model, -0.5686802931902314) >= 0.5):
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