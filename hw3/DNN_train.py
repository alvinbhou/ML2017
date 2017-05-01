import math, csv, random, copy
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import theano
import time
import os
os.environ["THEANO_FLAGS"] = "device=gpu0"



def loadData(data):
    X = []
    Y = []
    f = open('train.csv')    
    for row in csv.reader(f):       
        if(row[0] == 'label'):
            continue        
        y = int(row[0])
        x = np.array(row[1].split(" "))
        x = np.array(list(map(float, x))).reshape(2304)

        X.append(x)
        Y.append(y)
    X = np.array(X)
    X = X / 255 
    Y = keras.utils.to_categorical(Y,  num_classes = 7)
    return (X, Y)    
    
    pass


# def sampleData(seed, ratio, x_train, y_train):
#     dataLen = len(x_train)
#     d = copy.copy(data)
#     random.seed(seed)
#     random.shuffle(d)
#     train_data = d[:math.floor(dataLen*ratio)]
#     valid_data = d[math.floor(dataLen*ratio):]
#     return( (train_data, valid_data))


    
    
    

start_time = time.time()



print("Create Model")

def genModelandCompile(X_train, X_valid, y_train, y_valid):

    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    # model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu'))
    # # model.add(Conv2D(32, (3, 3), activation='relu'))
    
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(128, (2, 2), activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(128, (2, 2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))
    # model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    # model.add(Flatten())
    model.add(Dense(200, input_dim= 2304, activation = 'relu'))  
    model.add(Dropout(0.25))
    model.add(Dense(500, activation = 'relu'))
    model.add(Dropout(0.25))
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

    batchSize = 600
    epoch = 50
    curTime = time.time()
    model.fit(X_train, y_train, batch_size = batchSize,epochs = epoch)   


    scores = model.evaluate(X_valid, y_valid, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model.save('model/' + str(int(curTime))+ '_' + str(scores[1]*100)+ 'model.h5')
    json = model.to_json()
    with open('model/' + str(int(curTime))+ 'model.json', 'w') as json_file:
        json_file.write(str(tsize) + ' ' + str(rnState) + '\n')
        json_file.write("batchsize = " + str(batchSize) + "  epoch = " + str(epoch) + "\n")
        json_file.write(json)


data = []
X, Y = loadData(data)

for i in range(0,1):
    tsize = 0.06
    rnState = 42
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size= tsize, random_state=rnState * i)
    genModelandCompile(X_train, X_valid, y_train, y_valid)


print("--- %s seconds ---" % (time.time() - start_time))    
