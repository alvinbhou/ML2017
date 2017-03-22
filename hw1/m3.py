import csv, math
import time



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



def gradientDescent(iteration):	
	minError = 99999
	# ydata = b + w * xdata 
	b = 1 # initial b
	w = [initPara] * len(trainSet[0][1]) # initial w1 ... w9
	# w = [0.15463810310296658,-0.22334730272657,0.17318108677263902,-0.04106481017562704,-0.31665746035693665,0.6189104367365011,-0.48575299448433595,-0.21087341712830965,1.261847076351316]
	lr = 0.1 # learning rate
	# iteration = 100000

	b_lr = 0.0
	w_lr = [0.0] * len(trainSet[0][1])	

	# Store initial values for plotting.
	b_history.append(b)
	w_history.append(w)

	p = None 
	# Iterations
	for it in range(iteration):
	    
	    b_grad = 0.0
	    w_grad = [0.0]* 18
	    for n in range(len(trainSet)):
	    	# trainSet[n][0] = y_data
	    	# trainSet[n][1] = x_data list

	       	b_grad = b_grad  - 2.0*(trainSet[n][0] - b - sum([a*b for a,b in zip(w,trainSet[n][1])]) )*1.0
	       	for i in range(0,9): 
	        	w_grad[i] = w_grad[i]  - 2.0*(trainSet[n][0] - b - sum([a*b for a,b in zip(w,trainSet[n][1])]))*trainSet[n][1][i] + 2 * lamb * w[i]
	    
	    b_lr = b_lr + b_grad**2
	    # Update parameters.
	    b = b - lr/math.sqrt(b_lr) * b_grad 
	    for i in range(0,9):
	   		w_lr[i] = w_lr[i] + w_grad[i]**2
	   		w[i] = w[i] - lr/math.sqrt(w_lr[i]) * w_grad[i]   		
	    
	    # Store parameters for plotting
	    b_history.append(b)
	    w_history.append(w)			
	# for i in range(iteration):
	# 	print(w_history[i])
	# print([b_history[-1]] + w_history[-1])

	# return((b_history[-1], w_history[-1]))

def errorChecking(optModel, validSet):
	SE = 0
	for vs in validSet:
		# print(str(vs[0]) + " " + str(( optModel[0] + sum([a*b for a,b in zip(vs[1],optModel[1])]))))
		SE = SE + (vs[0] - ( optModel[0] + sum([a*b for a,b in zip(vs[1],optModel[1])])) ) ** 2
	MRSE = 	math.sqrt(SE/len(validSet))
	return(MRSE)

b_history = []
w_history = []
trainSet = []
validSet = []

start_time = time.time()

lamb = 0.5

print("Include more var?") # 1 for pM25 only, 2 for other
varNum = input()
# varNum = "3"

print("Load num of data:")
num = input()
# num = "13"
print("Load training or load all:")
mode = input()
# mode = "2"
initPara = loadData(varNum, int(mode), num) # 1 for first time training, 2 for load all
print(initPara)
print("Number of iteration")
iteration = int(input())
# iteration = 8000
gradientDescent(iteration)

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

print("Current MRSE:" + str(MRSE) + " " + str(it))
	



print("Check through direct id")
did = input()
print("Check through number of data?")
yes = input()

accError = 0.0

for i in range(1,int(yes)+1):	
	vset = []
	f = open('data' + did + '/valid'+ str(i) +'.csv', 'r')	
	for row in csv.reader(f):
		row = list(map(float, row))
		y_value = row[9]
		del row[9]				
		vset.append((float(y_value), row))
	err = errorChecking(optModel, vset)
	print(err)
	accError = accError + err
	# print(len(vset))

	f.close()
print("avg error:")
print(accError/(int(yes)))

with open("model3.csv", "a", newline='') as mFile:
	if(varNum == "1"):
		s = "PM25"
	elif(varNum == "2"):
		s = "MORE"
	elif(varNum == "3"):
		s = "BIGDATA25"
	mFile.write(str(MRSE)+ " " + num + " " + mode + " " + str(iteration) + " " + s + "lambda " + str(lamb))
	mFile.write('\n')

	writer = csv.writer(mFile)
	writer.writerow([optModel[0]]+optModel[1])

print("--- %s seconds ---" % (time.time() - start_time))



# errorChecking(optModel)








