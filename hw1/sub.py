import csv

model = [1.5346760698571749,-0.013273826349830549,-0.05214926357975244,0.41837890501358654,-0.33632825192521704,-0.3157005942550218,0.8236493192057989,-0.7632722186631162,0.10902448058134745,1.0687087729717102]

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