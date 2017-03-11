import csv, math



def loadData(para, num):
	if(para == 1):
		f = open('data/training'+ num +'.csv', 'r')
	else:
		f =  open('data/all'+ num +'.csv', 'r')

	tmpValue = [0.0] * 10
	tmpY = 0.0
	
	for row in csv.reader(f):
		row = list(map(float, row))
		s = (row[-1],row[0:9] )
		
		tmpValue = [x + y for x, y in zip(tmpValue, s[1])]
		tmpY = tmpY +  s[0]
		trainSet.append(s)

	tmpValue[:] = [(x / len(trainSet) ) for x in tmpValue]
	avgY = tmpY / len(trainSet)	
	avgPara = avgY/sum(tmpValue)	
	f.close()

	f = open('data/valid'+ num +'.csv', 'r')
	for row in csv.reader(f):
		row = list(map(float, row))
		validSet.append((row[-1], row[0:9]))
	f.close()

	return avgPara



def gradientDescent(iteration):	
	minError = 99999
	# ydata = b + w * xdata 
	b = 1 # initial b
	w = [initPara] * 9  # initial w1 ... w9
	# w = [0.15463810310296658,-0.22334730272657,0.17318108677263902,-0.04106481017562704,-0.31665746035693665,0.6189104367365011,-0.48575299448433595,-0.21087341712830965,1.261847076351316]
	lr = 0.1 # learning rate
	# iteration = 100000

	b_lr = 0.0
	w_lr = [0.0]* 9

	# Store initial values for plotting.
	b_history.append(b)
	w_history.append(w)

	p = None 

	# Iterations
	for it in range(iteration):
	    
	    b_grad = 0.0
	    w_grad = [0.0]*9
	    for n in range(len(trainSet)):
	    	# trainSet[n][0] = y_data
	    	# trainSet[n][1] = x_data list

	       	b_grad = b_grad  - 2.0*(trainSet[n][0] - b - sum([a*b for a,b in zip(w,trainSet[n][1])]) )*1.0
	       	for i in range(0,9): 
	        	w_grad[i] = w_grad[i]  - 2.0*(trainSet[n][0] - b - sum([a*b for a,b in zip(w,trainSet[n][1])]))*trainSet[n][1][i]
	    
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

print("Load num of data:")
num = input()
print("Load training or load all:")
mode = input()
initPara = loadData(int(mode), num) # 1 for first time training, 2 for load all
print("Number of iteration")
iteration = int(input())





gradientDescent(iteration)
# print(sol[0])
# print(sol[1])
# print(sol[2])
# print('\n')

minError = 9999
optModel = None
for i in range(iteration):
	m = (b_history[i],w_history[i])
	err = errorChecking(m,validSet)
	if(err < minError):
		minError = err
		optModel = m

MRSE = minError

print("Current MRSE:" + str(MRSE))


	

with open("model.csv", "a", newline='') as mFile:
	mFile.write(str(MRSE)+ " " + num)
	mFile.write('\n')
	writer = csv.writer(mFile)
	writer.writerow([optModel[0]]+optModel[1])

print("Check through number of data?")
yes = input()

accError = 0.0

for i in range(1,int(yes)+1):	
	vset = []
	f = open('data/valid'+ str(i) +'.csv', 'r')	
	for row in csv.reader(f):
		row = list(map(float, row))
		vset.append((row[-1], row[0:9]))
	err = errorChecking(optModel, vset)
	print(err)
	accError = accError + err
	# print(len(vset))

	f.close()
print("avg error:")
print(accError/(int(yes)))



# errorChecking(optModel)








