import os
import sys
import csv
import tensorflow as tf 
from keras.models import load_model
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
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
    X = np.array(X)
    X = X / 255 
    # Y = keras.utils.to_categorical(Y,  num_classes = 7)
    return X



model_name = "1493828072_65.0000000832model.h5"
emotion_classifier = load_model("theano_model/" + model_name)
emotion_classifier.summary()
layer_idx = [idx for idx, layer in enumerate(emotion_classifier.layers) if layer.name == "dense_2"][0]
private_pixels = loadData() 


input_img = emotion_classifier.input
img_ids = []

for i in range(1000,1020):
	img_ids.append(i)

for idx in img_ids:
	
	val_proba = emotion_classifier.predict(private_pixels[idx:idx+1])
	pred = val_proba.argmax(axis=-1)
	target = K.mean(emotion_classifier.output[:, pred])
	grads = K.gradients(target, input_img)[0]
	fn = K.function([input_img, K.learning_phase()], [grads])
	
	heatmap = visualize_saliency(emotion_classifier, layer_idx, pred, private_pixels[idx])

	threshold = 0.6
	see = private_pixels[idx].reshape(48, 48)
	mp = np.mean(see)
	for i in range(0,48):
		for j in range(0,48):
			if np.mean(heatmap[i][j]) < threshold:
				see[i][j] = mp

	if not os.path.exists("saliency_map/train/"+ model_name):
		os.makedirs("saliency_map/train/"+model_name)

	plt.figure()
	plt.imshow(heatmap, cmap=plt.cm.jet)
	plt.colorbar()
	plt.tight_layout()
	fig = plt.gcf()
	plt.draw()
	fig.savefig("saliency_map/train/"+model_name+"/"+str(idx)+".png", dpi=100)

	plt.figure()
	plt.imshow(see,cmap='gray')
	plt.colorbar()
	plt.tight_layout()
	fig = plt.gcf()
	plt.draw()
	fig.savefig("saliency_map/train/"+model_name+"/"+str(idx)+"_mask.png", dpi=100)
	plt.close()