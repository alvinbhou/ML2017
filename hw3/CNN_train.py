import math, csv, random, copy
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import theano
import time
import os
os.environ["THEANO_FLAGS"] = "device=gpu0"



def loadData(data, x_train, y_train):
    x_train = []
    y_train = []
    f = open('train.csv')    
    for row in csv.reader(f):       
        if(row[0] == 'label'):
            continue        
        y = int(row[0])
        x = np.array(row[1].split(" "))
        x = np.array(list(map(float, x))).reshape((48,48,1))

        x_train.append(x)
        y_train.append(y)
        data.append((x,y))
    x_train = np.array(x_train)
    x_train = x_train / 255 
    y_train = keras.utils.to_categorical(y_train,  num_classes = 7)
    return (x_train, y_train)    
    
    pass


def sampleData(seed, ratio, data):
    dataLen = len(data)
    d = copy.copy(data)
    random.seed(seed)
    random.shuffle(d)
    train_data = d[:math.floor(dataLen*ratio)]
    valid_data = d[math.floor(dataLen*ratio):]
    return( (train_data, valid_data))


    
    
    

start_time = time.time()

data = []
x_train = []
y_train = []

seed = 0
ratio = 0.1
x_train, y_train = loadData(data, x_train, y_train)


# print(x_train[0])



# (train_data, valid_data) = sampleData(seed, ratio, data)


# print(x_train[0])
print("Create Model")

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(25, (3, 3), input_shape=(48, 48, 1), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(100, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(200, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
# model.add(Dropout(0.25))
# model.add(Dense(80, activation = 'relu'))
# model.add(Dropout(0.25))
# model.add(Dense(100, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(88, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(96, activation = 'relu'))
# model.add(Dropout(0.2))
model.add(Dense(7, activation = 'softmax'))
model.summary()



print("Start compile")
model.compile(loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 500,epochs = 40)   

x_valid = x_train[20000:]
y_valid = y_train[20000:]
scores = model.evaluate(x_valid, y_valid, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save(str(int(start_time))+ '_' + str(scores[1]*100)+ 'model.h5')

# # serialize model to JSON
# model_json = model.to_json()
# with open("model6.json", "w") as json_file:
#     json_file.write(model_json)


print("--- %s seconds ---" % (time.time() - start_time))    
