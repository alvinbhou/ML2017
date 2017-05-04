import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
import numpy as np
import csv
from vis.utils import utils
from vis.visualization import visualize_saliency


def loadData():
    X = []
    Y = []
    f = open('train.csv')  
    num = 0  
    for row in csv.reader(f):       
        if(row[0] == 'label'):
            continue       
      
        x = np.array(row[1].split(" "))
        x = np.array(list(map(float, x))).reshape((48,48,1))
        X.append(x)
       
        num = num + 1
        if(num >= 1020):
            break
    X = np.array(X)
    X = X / 255 
    # Y = keras.utils.to_categorical(Y,  num_classes = 7)
    return X

model_name = "1493828072_65.0000000832model.h5"
emotion_classifier = load_model("theano_model/" + model_name)
emotion_classifier.summary()


layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[0:])


input_img = emotion_classifier.input
name_ls = ['conv2d_1']
collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]


private_pixels = loadData() 
# private_pixels = load_pickle('../fer2013/privateTest_pixels.pkl')
private_pixels = [ private_pixels[i].reshape((1, 48, 48, 1)) 
                    for i in range(len(private_pixels)) ]
print(private_pixels[0].shape)
choose_id = 1019
photo = private_pixels[choose_id]
for cnt, fn in enumerate(collect_layers):
    im = fn([photo, 0]) #get the output of that layer   
    fig = plt.figure(figsize=(14, 8))
    nb_filter = im[0].shape[3]
    for i in range(nb_filter):
        ax = fig.add_subplot(nb_filter/8, 8, i+1)
        ax.imshow(im[0][0, :, :, i], cmap='BuGn')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
    img_path = "./report"
    if not os.path.isdir(img_path):
        os.mkdir(img_path)
    fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))