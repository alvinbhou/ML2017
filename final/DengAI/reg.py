import sys
import csv
import numpy as np

from keras.optimizers import Adam
from keras.layers import Dense,Dropout
from keras.layers import GRU, LSTM
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import keras.backend as K 

# Load data
def loadXData(path):
    xData = []
    f = open(path, 'r')
    rawData = csv.reader(f, delimiter=',')
    rawData.__next__()
    for row in rawData:
        xData.append([(0 if row[0] == 'sj' else 1), int(row[2])] + [float(x) if x != '' else 0 for x in row[4:]])
    f.close()
    xData = np.array(xData)
    return xData

def loadYdata(path):
    yData = []
    f = open(path, 'r')
    rawData = csv.reader(f, delimiter=',')
    rawData.__next__()
    for row in rawData:
        yData.append(float(row[3]))
    f.close()
    yData = np.array(yData)
    return yData


# def repeatData(data, nTimes):
#     result = [()]
    

trainXData = loadXData('dengue_features_train.csv')
trainYData = loadYdata('dengue_labels_train.csv')



# howMany = (int)(len(trainXData)*0.9)
# validXData = trainXData[howMany:]
# validYData = trainYData[howMany:]
# trainXData = trainXData[:howMany]
# trainYData = trainYData[:howMany]

# print(validXData)
# print(validYData)


dimention = len(trainXData[0])

print(dimention)

def baseline_model():
    model = Sequential()
    # model.add(GRU(64,activation='tanh',dropout=0, input_dim=dimention, input_length=1)) #128
    model.add(Dense(200, input_dim=dimention, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100,activation='relu')) #256
    # model.add(Dropout(0.1))
    model.add(Dense(50,activation='relu')) #256
    # model.add(Dropout(0.1))
    # model.add(Dense(150,activation='relu')) #256
    # model.add(Dropout(0.4))
    model.add(Dense(1, kernel_initializer='normal')) #softmax
    model.summary()

    adam = Adam(lr=0.02,decay=1e-6,clipvalue=0.5) # 0.002
    model.compile(loss='mean_squared_error',
                    optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# scale = StandardScaler()
# trainXData = scale.fit_transform(trainXData)
# trainYData = scale.fit_transform(trainYData)


print(trainXData)
print(trainYData)

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=1000, batch_size=5, verbose=0)

# estimator.fit(trainXData, trainYData)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, trainXData, trainYData, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# hist = model.fit(trainXData, trainYData, 
#                      validation_data=(validXData, validYData),
#                      epochs=200, 
#                      batch_size=32)

