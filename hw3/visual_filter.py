import matplotlib.pyplot as plt
import math, csv, random, copy
import numpy as np 
import keras
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import load_model
import os
from keras import applications
from keras import backend as K
from scipy.misc import imsave


name = '1493828072_65.0000000832model.h5'
model = load_model('theano_model/' + name)
model.summary()

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])
print(layer_dict)
layer_name = 'conv2d_1'
filter_index = 31
layer_output = layer_dict[layer_name].output

nb_filter = 32
fig = plt.figure(figsize=(14, 8))
for i in range(nb_filter):
    loss = K.mean(layer_output[:, :, :, i])
    input_img = model.input
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)


    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])
    input_img_data = np.random.random((1, 48, 48, 1)) # random noise

    # run gradient ascent for 1000 steps
    for step in range(1000):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value 

    img = input_img_data[0]
    img = img.reshape(48,48)
    
    ax = fig.add_subplot(nb_filter/8, 8, i+1)
    ax.imshow(img, cmap='BuGn')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel(str(i))
    plt.tight_layout()

fig.show()
fig.suptitle('Filters of layer conv2d_1')
fig.savefig('visual_filter.png') #將圖片儲存至disk

