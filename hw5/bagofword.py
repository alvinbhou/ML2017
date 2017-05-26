from __future__ import print_function

import numpy as np
import keras, sys
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
import keras.backend as K 
from keras.callbacks import EarlyStopping, ModelCheckpoint
#####################
###   parameter   ###
#####################
split_ratio = 0.05
# nb_epoch = 1000
batch_size = 200

max_words = 100000
epochs = 100


train_path = sys.argv[1]
test_path = sys.argv[2]

def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)

    if tp == 0:
        return 0
    
    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))

print('Loading data...')
# start_time = int(time.time())
### read training and testing data
(Y_data,X_data,tag_list) = read_data(train_path,True)
(_, X_test,_) = read_data(test_path,False)
all_corpus = X_data + X_test
print ('Find %d articles.' %(len(all_corpus)))

### tokenizer for all data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_corpus)
word_index = tokenizer.word_index


### convert word sequences to index sequence
print ('Convert to index sequences.')
train_sequences = tokenizer.texts_to_sequences(X_data)
test_sequences = tokenizer.texts_to_sequences(X_test)

# ### padding to equal length
# print ('Padding sequences.')
# train_sequences = pad_sequences(train_sequences)
# max_article_length = train_sequences.shape[1]
# test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)

train_sequences = np.array(train_sequences)
test_sequences = np.array(test_sequences)

###
train_tag = to_multi_categorical(Y_data,tag_list) 

### split data into training set and validation set
(X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)

print(len(X_train), 'train sequences')
print(len(X_val), 'test sequences')
# print(y_train[0])

num_classes = 38
print(num_classes, 'classes')

###
train_tag = to_multi_categorical(Y_data,tag_list) 




# print(X_val[0])

tokenizer = Tokenizer(num_words=max_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_val = tokenizer.sequences_to_matrix(X_val, mode='binary')
print('x_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)




# print('Convert class vector to binary class matrix '
#       '(for use with categorical_crossentropy)')
# Y_train = keras.utils.to_categorical(Y_train, num_classes)
# Y_val = keras.utils.to_categorical(Y_val, num_classes)
print('y_train shape:', Y_train.shape)
print('Y_val shape:', Y_val.shape)

# for x in X_train:
#     for r in x:
#         if r > 1:
#             print (r)


print('Building model...')
model = Sequential()
model.add(Dense(720, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(360,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(180,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(90,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))
model.summary()
opt = Adam(lr=0.002)
model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=[f1_score])

earlystopping = EarlyStopping(monitor='val_f1_score', patience = 20, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath= 'best.h5',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True,
                                monitor='val_f1_score',
                                mode='max')
hist = model.fit(X_train, Y_train, 
                    validation_data=(X_val, Y_val),
                    epochs=epochs, 
                    batch_size=batch_size,
                    callbacks=[earlystopping,checkpoint])

# history = model.fit(X_train, Y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_split=0.1)

score = model.evaluate(X_val, Y_val,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])