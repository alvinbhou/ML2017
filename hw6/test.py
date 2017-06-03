import numpy as np
import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = np.array([[2,5,6,0],[0,1,0,1]])
b = np.array([[2,5,-1,6], [0,2,0,2]])

input_a = keras.layers.Input(shape = [4])
input_b = keras.layers.Input(shape = [4])



out = keras.layers.Concatenate(axis = -1)([input_a, input_b])

model = keras.models.Model([input_a, input_b], out)

print(model.predict([a,b]))