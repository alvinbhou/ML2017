import csv, copy

model = [1.9462370098435744,-0.0004923569838956585,-0.08906807016255172,0.31643327840311236,-0.3147588879354323,-0.07701485748085794,0.6230676918036666,-0.6728575524359923,-0.00742618741397945,1.1433061517098675]


class DayData(object):
    def __init__(self, id):
        self.id = id
        self.PM25 = [0]*9
        self.PM10 = [0]*10

data = []
y = [0] * 240
def filterData():
	f = open('test_X.csv', 'r')	
	if(more == "1"):
		for row in csv.reader(f):				
			if(row[1] == "PM2.5"):	
				results = list(map(float, row[2:]))
				data.append(results)	
	else:
		d = DayData(0)
		day = 1
		for row in csv.reader(f):		
			row[0] = row[0][3:] # get day number		
			if(d.id != int(row[0])):
				tmp = copy.copy(d)
				data.append(tmp)
				d.id = int(d.id) + 1
				day = day + 1		
			if(row[1] == "PM10"):
				results = list(map(float, row[2:]))
				d.PM10 = results				
			elif(row[1] == "PM2.5"):		
				results = list(map(float, row[2:]))
				d.PM25 = results				
		data.append(d)	
	f.close()

print("Only PM25 or more?")
more = input()

filterData()


# for i in range(len(data)):
# 	print(data[i].id)
# 	print(data[i].PM10)
# 	print(data[i].PM25)

b = model[0]
model = model[1:]

if(more == "1"):
	for i in range(0,len(data)):	
		y[i] = b + sum([a*b for a,b in zip(model,data[i])])
		if(y[i] <= 0):
			y[i] = 0
		print(y[i])
else:
	for i in range(0,len(data)):
		para = data[i].PM25 + data[i].PM10
		y[i] = b + sum([a*b for a,b in zip(model,para)])
		if(y[i] <= 0):
			y[i] = 0
		print(y[i])

with open("result.csv", "w", newline='') as mFile:
	writer = csv.writer(mFile)
	writer.writerow(["id","value"])
	for i in range(0, len(data)):
		mFile.write("id_" + str(i) + ",")
		mFile.write(str(y[i]))
		mFile.write("\n")