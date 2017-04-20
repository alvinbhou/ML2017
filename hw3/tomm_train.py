import time
import csv
import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
os.environ["THEANO_FLAGS"] = "device=gpu0"

#categorical_crossentropy

def load_data(fileX):
        train_raw = list(csv.reader(open(fileX,'r')))
        train_num = 20000
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        print ("Begin loading...")
        for item in train_raw[1:]:
                y_train.append(int(item[0]))
                x_train.append([int(i) for i in item[1].split(" ")])
        print ("Loading completed")

        x_test = np.array(x_train[train_num:])
        y_test = np.array(y_train[train_num:])
        x_train = np.array(x_train[:train_num])
        y_train = np.array(y_train[:train_num])

        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train = x_train.reshape(len(x_train), 48*48)
        # x_test = x_test.reshape(len(x_test), 48*48)
        x_train = x_train.reshape(len(x_train),48,48,1)
        x_test = x_test.reshape(len(x_test),48,48,1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, 7)
        y_test = np_utils.to_categorical(y_test, 7)
        x_train = x_train/255
        x_test = x_test/255
        #x_test=np.random.normal(x_test)
        return (x_train, y_train), (x_test, y_test)

(x_train,y_train),(x_test,y_test)=load_data("train.csv")

model = Sequential()
model.add(Conv2D(25,(3,3),input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(50,(2,2)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(100,(2,2)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
# model.add(Conv2D(200,(2,2)))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=400,activation='relu'))
model.add(Dense(units=7,activation='softmax'))
model.summary()



model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=200,epochs=25)

model.save("CNN_1.h5")
score = model.evaluate(x_train,y_train)
print ('\nLoss:' +str (score[0]))
print ('\nTrain Acc:'+ str (score[1]))
score = model.evaluate(x_test,y_test)
print ('\nLoss:' +str(score[0]))
print ('\nTest Acc:'+str( score[1]))


