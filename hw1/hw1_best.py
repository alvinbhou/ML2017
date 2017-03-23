import csv, math
import time,sys

# function loading training data
def loadData(varNum, para, num):
	if(para == 1):
		f = open('data' + varNum + '/training'+ num +'.csv', 'r')
	else:
		f =  open('data' + varNum + '/all'+ num +'.csv', 'r')
	for row in csv.reader(f):
		rowLen = len(row)
		break
	tmpValue = [0.0] * rowLen
	tmpY = 0.0	
	
	for row in csv.reader(f):
		row = list(map(float, row))
		y_value = row[9]
		del row[9]
		s = (y_value, row)
		
		tmpValue = [x + y for x, y in zip(tmpValue, s[1])]
		tmpY = tmpY +  s[0]
		trainSet.append(s)

	tmpValue[:] = [(x / len(trainSet) ) for x in tmpValue]
	avgY = tmpY / len(trainSet)	
	avgPara = avgY/sum(tmpValue)

	f.close()

	f = open('data' + varNum + '/valid'+ num +'.csv', 'r')
	for row in csv.reader(f):
		y_value = row[9]
		del row[9]
		row = list(map(float, row))
		validSet.append((float(y_value), row))
	f.close()
	return avgPara

# gradient descent 
def gradientDescent(iteration):	
	minError = 99999
	b = 1 # initial b
	w = [initPara] * len(trainSet[0][1]) # initial w1 ... w9
	lr = 0.1 # learning rate

	b_lr = 0.0
	w_lr = [0.0] * len(trainSet[0][1])	

	# Store initial values for plotting.
	b_history.append(b)
	w_history.append(w)

	# Iterations
	for it in range(iteration):
	    
	    b_grad = 0.0
	    w_grad = [0.0]* 18
	    for n in range(len(trainSet)):
	       	b_grad = b_grad  - 2.0*(trainSet[n][0] - b - sum([a*b for a,b in zip(w,trainSet[n][1])]) )*1.0
	       	for i in range(0,9): 
	        	w_grad[i] = w_grad[i]  - 2.0*(trainSet[n][0] - b - sum([a*b for a,b in zip(w,trainSet[n][1])]))*trainSet[n][1][i]
	    
	    b_lr = b_lr + b_grad**2
	    # Update parameters.
	    b = b - lr/math.sqrt(b_lr) * b_grad 
	    for i in range(0,9):
	   		w_lr[i] = w_lr[i] + w_grad[i]**2
	   		w[i] = w[i] - lr/math.sqrt(w_lr[i]) * w_grad[i]   		
	    
	    # Store parameters 
	    b_history.append(b)
	    w_history.append(w)			

# compute MRSE with validation set
def errorChecking(optModel, validSet):
	SE = 0
	for vs in validSet:
		SE = SE + (vs[0] - ( optModel[0] + sum([a*b for a,b in zip(vs[1],optModel[1])])) ) ** 2
	MRSE = 	math.sqrt(SE/len(validSet))
	return(MRSE)


def filterData():
	f = open(inputFile2, 'r')	
	for row in csv.reader(f):				
		if(row[1] == "PM2.5"):	
			results = list(map(float, row[2:]))
			data.append(results)	
	f.close()

def sub(model):
	b = model[0]
	model = model[1:]
	for i in range(0,len(data)):	
		y[i] = b + sum([a*b for a,b in zip(model,data[i])])
		if(y[i] <= 0):
			y[i] = 0

	with open(outputFile, "w", newline='') as mFile:
		writer = csv.writer(mFile)
		writer.writerow(["id","value"])
		for i in range(0, len(data)):
			mFile.write("id_" + str(i) + ",")
			mFile.write(str(y[i]))
			mFile.write("\n")

b_history = []
w_history = []
trainSet = []
validSet = []
data = []
y = [0] * 240


inputFile1 = sys.argv[1]
inputFile2 = sys.argv[2]
outputFile = sys.argv[3]
# start_time = time.time()

# sample data #3 directory, pure PM2.5 data
varNum = "3"
# num 13th training data
num = "13"
# 2 for load all
mode = "2"

# iteration 
iteration = 30000

# load data and do gradient descent
'''
initPara = loadData(varNum, int(mode), num) # 1 for first time training, 2 for load all
gradientDescent(iteration)
'''

# get the optimal model
'''
minError = 9999
optModel = None
it = 0
for i in range(iteration):
	m = (b_history[i],w_history[i])
	err = errorChecking(m,validSet)
	if(err < minError):
		minError = err
		optModel = m
		it = i
MRSE = minError
'''

# load arguments from model
m = open('model_best', 'r')
for row in csv.reader(m):
	model = row
# model = [optModel[0]] + optModel[1]
model = list(map(float, model))

# load and filter PM2.5 data
filterData()

#submit the answer
sub(model)










