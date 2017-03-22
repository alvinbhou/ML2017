import csv

def loadTrainData(dataNum):
	f = open('data3/training'+ str(dataNum) +'.csv', 'r')
	for row in csv.reader(f):
		results = list(map(float, row))
		data.append(results)
	f.close()
	main  = open('PMdata.csv', 'r')
	for row in csv.reader(main):
		if(row[1] == "PM2.5"):
			row = list(map(float,row[2:]))
			if(-1 not in row):
				mData.append(row)


def findIndex():
	day = 0
	for s in data:	
		index = [(i, i+len(s)) for i in range(len(mData[day])) if mData[day][i:i+len(s)] == s]		
		if(len(index) == 0):
			day = day + 1
			index = [(i, i+len(s)) for i in range(len(mData[day])) if mData[day][i:i+len(s)] == s]
		dataIndex.append((day,index[0][0]))


data = []
mData = []
dataIndex = []

loadTrainData(14)
findIndex()



a = [1,4,5,6,7,8,4]
b = [4,5,6]

f = open('index.csv','w')
for index in dataIndex:
	f.write(str(index[0]) + ',' + str(index[1]) + '\n')	


