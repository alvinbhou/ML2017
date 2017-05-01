import math, csv, random, copy
import numpy as np 
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import theano
import time
import os
# os.environ["THEANO_FLAGS"] = "device=gpu0"



def loadData():
    X = []
    Y = []
    f = open('train.csv')  
    num = 0  
    for row in csv.reader(f):       
        if(row[0] == 'label'):
            continue       
        if(num in badImg):
            num = num + 1
            continue    
        y = int(row[0])
        x = np.array(row[1].split(" "))
        x = np.array(list(map(float, x))).reshape((48,48,1))

        X.append(x)
        Y.append(y)
        num = num + 1
    X = np.array(X)
    X = X / 255 
    Y = keras.utils.to_categorical(Y,  num_classes = 7)
    return (X, Y)    
    
  
    
    

start_time = time.time()



print("Create Model")

def genModelandCompile(X_train, X_valid, y_train, y_valid):

    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), input_shape=(48, 48, 1), activation='relu'))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))   
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))   
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(40, activation = 'relu'))
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
  
   
    model.fit(X_train, y_train, batch_size = batchSize,epochs = epoch)   
    scores = model.evaluate(X_valid, y_valid, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model ,scores


def saveModel(model, tsize, rnState, scores):


    mode = 'tensorflow_'
    curTime = time.time()
    model.save(mode + 'model/' + str(int(curTime))+ '_' + str(scores[1]*100)+ 'model.h5')
    json = model.to_json()
    with open(mode + 'model/' + str(int(curTime))+ 'model.json', 'w') as json_file:
        json_file.write(str(tsize) + ' ' + str(rnState) + '\n')
        json_file.write("batchsize = " + str(batchSize) + "  epoch = " + str(epoch) + "\n")
        json_file.write(json)

badImg = [59,2171,2809,4275,5274,5439,5881,6102, 6458,7172,7496,7527,7629,8423,8737,9026,9500,9679,9797,10423,10657,11244,11286,11295,11846,12352,13988,14279,15144,15835,15838,15894,16540,19238,19632,20712,20817,21817,22198,22927,23596,23894,24053,24441,25909,26383,26860,26897]


X, Y = loadData()
print("data loaded")
batchSize = 600
epoch = 1

tsize = 0.06
rnState = 0
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size= tsize, random_state=rnState)
model, scores = genModelandCompile(X_train, X_valid, y_train, y_valid)

saveModel(model, tsize, rnState , scores)



print("--- %s seconds ---" % (time.time() - start_time))    
