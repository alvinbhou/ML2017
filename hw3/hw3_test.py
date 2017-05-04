import math, csv, random, copy,sys
import numpy as np 
import keras
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import load_model
import os

def loadTestData(test_file, x_test):
    f = open(test_file)
    for row in csv.reader(f):       
        if(row[0] == 'id'):
            continue        
        x = np.array(row[1].split(" "))
        x = np.array(list(map(int, x))).reshape((48,48,1))
        x_test.append(x)           
    pass

model_name = 'model_65.h5'
model = load_model(model_name)
# model.summary()

test_file = sys.argv[1]



x_test = []
loadTestData(test_file,x_test)
x_test = np.array(x_test)
x_test = x_test / 255

result = model.predict(x_test)
result_file = sys.argv[2]

with open(result_file , "w", newline='') as mFile:
    writer = csv.writer(mFile)
    writer.writerow(["id","label"])
    for i in range(0, len(result)):
        mFile.write(str(i) + ",")       
        mFile.write(str(np.argmax(result[i])))
        mFile.write("\n")

