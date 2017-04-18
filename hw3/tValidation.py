import math, csv, random, copy
import numpy as np 
import keras
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
loadData(data, x_train, y_train)
x_train = np.array(x_train)

y_train = keras.utils.to_categorical(y_train,  num_classes = 7)
# load weights into new model
model = load_model("c_try.h5")
print("Loaded model from disk")

x_test = []
loadTestData(x_test)
x_test = np.array(x_test)

result = model.predict(x_test)

with open('result.csv', "w", newline='') as mFile:
    writer = csv.writer(mFile)
    writer.writerow(["id","label"])
    for i in range(0, len(result)):
        mFile.write(str(i) + ",")
        max_value = max(result[i])
        max_index = row.index(max_value)
        mFile.write(str(y_pred[i]))
        mFile.write("\n")


print(result[0])
print(sum(result[0]))
print(len(result))
 
# evaluate loaded model on test data
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# scores = model.evaluate(x_train, y_train, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))