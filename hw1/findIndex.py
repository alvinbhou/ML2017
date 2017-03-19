import csv

def loadTrainData(dataNum):
	f = open('data1/training'+ dataNum +'.csv', 'r')
	for row in csv.reader(f):
		results = list(map(float, row))
		data.append(results)
	f.close()
	main  = open('PMdata.csv', 'r')
	for row in csv.reader(main):
		if(row[1] == "PM2.5"):
			row = row[2:]


data = []


a = [1,4,5,6,7,8,4]
b = [4,5,6]


