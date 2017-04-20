import math, csv, random
import numpy as np 


def loadData(data):
    f = open('train.csv')    
    for row in csv.reader(f):
        y = row[0]
        x = row[1].split(" ")
        data.append((x,y))

def sampleData(data,seed,ratio):
    dataLen = len(data)
	d = copy.copy(data)
	random.seed(seed)
	random.shuffle(d)
	train_data = d[:math.floor(dataLen*ratio)]
	valid_data = d[math.floor(dataLen*ratio):]
	return(train_data, valid_data)



data = []
seed = 0
ratio = 0.1







loadData(data)
(train_data, valid_data) = sampleData(data,seed,ratio)

print(data[2])


