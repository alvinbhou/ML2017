import csv, numpy
f = open('train.csv', 'r', encoding="big5")
dataAll = []
day = -1
for row in csv.reader(f):
   	del row[0]
   	if(row[1] == "AMB_TEMP"):
   		day = day + 1    
   	if(row[1] == "RAINFALL"):
   		for i in range(0,len(row)):
   			if(row[i] == "NR"):
   				row[i] = 0

   	row[0] = "d" + str(day)
   	dataAll.append(row)
   	
	

with open("PMdata.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(dataAll)


    	

