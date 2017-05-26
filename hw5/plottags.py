import numpy as np
import string
import sys
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import pickle
import operator
import matplotlib.pyplot as plt
plt.rcdefaults()



train_path = sys.argv[1]
test_path = sys.argv[2]


#####################
###   parameter   ###
#####################
split_ratio = 0.05
embedding_dim = 200
nb_epoch = 1000
batch_size = 200



################
###   Util   ###
################
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

#########################
###   Main function   ###
#########################
def main():
    start_time = int(time.time())
    ### read training and testing data
    (Y_data,X_data,tag_list) = read_data(train_path,True)
    (_, X_test,_) = read_data(test_path,False)
    all_corpus = X_data + X_test
    print ('Find %d articles.' %(len(all_corpus)))

    count = np.zeros(38)
    for y_tags in Y_data:
        for y_tag in y_tags:
            count[tag_list.index(y_tag)] += 1 

    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    # Example data      
  
    y_pos = np.arange(38)
    ax.barh(y_pos, count, align='center', color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tag_list)
    ax.invert_yaxis()  # labels read top-to-bottom 

    fig.savefig('count.png')

    
    for idx, y_tags in enumerate(Y_data):
        y = np.sort(y_tags)
        y = y.flatten()
        print(y)
        Y_data[idx] = y
    d = dict()

    for y_tags in Y_data:  
        str1 = ' '.join(y_tags)      
        if str1 in d:
            d[str1] += 1
        else:
            d[str1] = 1
  
    sorted_x = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    
    print(sorted_x)
    sorted_x = sorted_x[0:30]
    x = []
    y = []
    for s in sorted_x:
        x.append(s[0])
        y.append(s[1])
    
    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.set_size_inches(24, 12)
    # Example data      
  
    y_pos = np.arange(30)
    ax.barh(y_pos, y, align='center', color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(x)
    ax.invert_yaxis()  # labels read top-to-bottom 

    fig.savefig('count2.png')
    

 
  
    
   

   

if __name__=='__main__':    
        main()