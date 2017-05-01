import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import csv
import keras
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import os

# base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# img_dir = os.path.join(base_dir, 'image')
# if not os.path.exists(img_dir):
#     os.makedirs(img_dir)
# cmap_dir = os.path.join(img_dir, 'cmap')
# if not os.path.exists(cmap_dir):
#     os.makedirs(cmap_dir)
# partial_see_dir = os.path.join(img_dir,'partial_see')
# if not os.path.exists(partial_see_dir):
#     os.makedirs(partial_see_dir)
# model_dir = os.path.join(base_dir, 'model')

def main():
    # parser = argparse.ArgumentParser(prog='plot_saliency.py',
    #         description='ML-Assignment3 visualize attention heat map.')
    # parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=80)
    # args = parser.parse_args()
    name = '1493568655_63.6680208903model.h5'    
    model = load_model('model/' + name)
    data = []
    X, y = loadData(data)
    
    # print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))

    private_pixels = load_pickle('../fer2013/privateTest_pixels.pkl')
    private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) 
                       for i in range(len(private_pixels)) ]
    input_img = model.input
    print(input_img)
    img_ids = [0]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(private_pixels[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        heatmap = None
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''

        thres = 0.5
        see = private_pixels[idx].reshape(48, 48)
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, 'privateTest', '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, 'privateTest', '{}.png'.format(idx)), dpi=100)

if __name__ == "__main__":
    main()