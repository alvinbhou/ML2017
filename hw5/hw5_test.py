import numpy as np
import string
import sys, pickle
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


import time, pickle

#####################
###   parameter   ###
#####################
split_ratio = 0.05
embedding_dim = 200
nb_epoch = 1000
batch_size = 200
num_words = 51867

train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]


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

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)

    if tp == 0:
        return 0
    
    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))

start_time = int(time.time())
### read training and testing data
(Y_data,X_data,tag_list) = read_data(train_path,True)
(_, X_test,_) = read_data(test_path,False)
all_corpus = X_data + X_test
print ('Find %d articles.' %(len(all_corpus)))

### tokenizer for all data
# tokenizer = Tokenizer()
tokenizer = pickle.load( open( 'tk.p', "rb" ) )

# tokenizer.fit_on_texts(all_corpus)
word_index = tokenizer.word_index
num_words = len(word_index) + 1

### convert word sequences to index sequence
print ('Convert to index sequences.')
train_sequences = tokenizer.texts_to_sequences(X_data)
test_sequences = tokenizer.texts_to_sequences(X_test)

### padding to equal length
print ('Padding sequences.')
train_sequences = pad_sequences(train_sequences)
max_article_length = train_sequences.shape[1]
test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)

print ('Get embedding dict from glove.')
# embedding_dict = get_embedding_dict('data/glove.6B.%dd.txt'%embedding_dim)
# print ('Found %s word vectors.' % len(embedding_dict))

# print ('Create embedding matrix.')
# embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

embedding_matrix = pickle.load( open( 'em.p', "rb" ) )



model = Sequential()
model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_article_length,
                    trainable=False))
model.add(GRU(360,activation='tanh',dropout=0.4))
model.add(Dense(720,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(360,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(180,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(38,activation='sigmoid'))
model.summary()

# adam = Adam(lr=0.002,decay=1e-6,clipvalue=0.5)
# model.compile(loss='categorical_crossentropy',
#                 optimizer=adam,
#                 metrics=[f1_score])

model_path = 'model.h5'
model.load_weights(model_path)

Y_pred = model.predict(test_sequences)
thresh = 0.4
with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)