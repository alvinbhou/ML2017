import csv

model = [1.1793149766961488,-0.0032884166541474113,-0.09952629681150872,0.3215011560656551,-0.3904099449876482,-0.007100584802389109,0.6452465437780649,-0.6310180767681093,-0.015823822036911298,1.1227866754788858]

data = []
y = [0] * 240
def filterData():
	f = open('test_X.csv', 'r')
	day = 1		
	for row in csv.reader(f):	
		if(row[1] == "PM2.5"):	
			results = list(map(float, row[2:]))
			data.append(results)	
	f.close()

filterData()
b = model[0]
model = model[1:]

for i in range(0,len(data)):	
	y[i] = b + sum([a*b for a,b in zip(model,data[i])])
	print(y[i])

with open("result.csv", "w", newline='') as mFile:
	writer = csv.writer(mFile)
	writer.writerow(["id","value"])
	for i in range(0, len(data)):
		mFile.write("id_" + str(i) + ",")
		mFile.write(str(y[i]))
		mFile.write("\n")