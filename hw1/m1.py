import csv, math



def loadData(para):
	if(para == 1):
		f = open('training.csv', 'r')
	else:
		f =  open('all.csv', 'r')

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

	f = open('valid.csv', 'r')
	for row in csv.reader(f):
		row = list(map(float, row))
		validSet.append((row[-1], row[0:9]))
	f.close()

	return avgPara



def gradientDescent(iteration):	
	minError = 99999
	# ydata = b + w * xdata 
	b = 0 # initial b
	w = [initPara] * 9  # initial w1 ... w9
	lr = 0.5 # learning rate
	# iteration = 100000

	b_lr = 0.0
	w_lr = [0.0]* 9

	# Store initial values for plotting.
	b_history = [b]
	w_history = [w]

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
	    if(it % 200 == 0):
	    	error = errorChecking((b_history[it-1],w_history[-1]))
	    	print(error)
	    	if(minError > error):
	    		minError = error
	    		optModel = (b_history[it-1],w_history[-1])
	return (minError, optModel)


	    		

			
	# for i in range(iteration):
	# 	print(w_history[i])
	# print([b_history[-1]] + w_history[-1])

	# return((b_history[-1], w_history[-1]))

def errorChecking(optModel):
	SE = 0
	for vs in validSet:
		# print(str(vs[0]) + " " + str(( optModel[0] + sum([a*b for a,b in zip(vs[1],optModel[1])]))))
		SE = SE + (vs[0] - ( optModel[0] + sum([a*b for a,b in zip(vs[1],optModel[1])])) ) ** 2
	MRSE = 	math.sqrt(SE/len(validSet))
	return(MRSE)


trainSet = []
validSet = []
initPara = loadData(1) # 1 for first time training, 2 for load all



sol = gradientDescent(30000)
MRSE = sol[0]
optModel = sol[1]

with open("model.csv", "a", newline='') as mFile:
	mFile.write(str(MRSE))
	mFile.write('\n')
	writer = csv.writer(mFile)
	writer.writerow([optModel[0]]+optModel[1])

# errorChecking(optModel)








