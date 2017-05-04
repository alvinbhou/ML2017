import itertools
import numpy as np
import matplotlib.pyplot as plt
import csv
import keras
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
from keras.models import load_model
from keras.utils import plot_model

import os

def loadData(data, id):
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
    # Y = keras.utils.to_categorical(Y,  num_classes = 7)
    return (X, Y)    

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# import some data to play with
data = []
badImg = [59,2171,2809,4275,5274,5439,5881,6102, 6458,7172,7496,7527,7629,8423,8737,9026,9500,9679,9797,10423,10657,11244,11286,11295,11846,12352,13988,14279,15144,15835,15838,15894,16540,19238,19632,20712,20817,21817,22198,22927,23596,23894,24053,24441,25909,26383,26860,26897]

X, y = loadData(data, 10)
class_names = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Split the data into a training set and a test set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size= 0.03, random_state=0)
# name = '1493568655_63.6680208903model.h5'
model = load_model('model_65.h5')
plot_model(model, to_file='model.png')
y_pred = model.predict(X_valid)
# model = load_model('model/' + name)
np.set_printoptions(precision=2)    
predictions = model.predict_classes(X_valid)
te_labels = y_valid

conf_mat = confusion_matrix(te_labels,predictions)


plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()