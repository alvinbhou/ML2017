import csv

model = [1.0225505954036065,-0.007675551725704934,-0.05295608726107858,0.2768588257628767,-0.2964053391261084,-0.06308222260724307,0.5532668298134261,-0.6262951543755991,0.02912954093874804,1.128972027395469]

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