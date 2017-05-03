import math, csv, random, copy
import numpy as np 
import keras
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import load_model
import os

def loadData(data, x_train, y_train):
    f = open('train.csv')    
    for row in csv.reader(f):       
        if(row[0] == 'label'):
            continue        
        y = int(row[0])
        x = np.array(row[1].split(" "))
        x = np.array(list(map(int, x))).reshape((48,48,1))

        x_train.append(x)
        y_train.append(y)
        data.append((x,y))       
    pass

def loadTestData(x_test):
    f = open('test.csv')
    for row in csv.reader(f):       
        if(row[0] == 'id'):
            continue        
        x = np.array(row[1].split(" "))
        x = np.array(list(map(int, x))).reshape((48,48,1))

        x_test.append(x)           
    pass

data = []
x_train = []
y_train = []

seed = 0
ratio = 0.1
# loadData(data, x_train, y_train)
# x_train = np.array(x_train)
# x_train = x_train / 255

# y_train = keras.utils.to_categorical(y_train,  num_classes = 7)
# load weights into new model
name = '1493755856_62.7906976744model.h5'
model = load_model('tensorflow_model/' + name)
model.summary()


print("Loaded model from disk")

x_test = []
loadTestData(x_test)
x_test = np.array(x_test)
x_test = x_test / 255

result = model.predict(x_test)

with open(name + 'result.csv', "w", newline='') as mFile:
    writer = csv.writer(mFile)
    writer.writerow(["id","label"])
    for i in range(0, len(result)):
        mFile.write(str(i) + ",")       
        mFile.write(str(np.argmax(result[i])))
        mFile.write("\n")



 
# evaluate loaded model on test data

# x_valid = x_train[10000:20000]
# y_valid = y_train[10000:20000]
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# scores = model.evaluate(x_train, y_train, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))