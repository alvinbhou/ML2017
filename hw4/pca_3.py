import numpy as np
from PIL import Image
from string import ascii_uppercase
import matplotlib.pyplot as plt

def loadData():
    data = []
    dirPath = 'faceExpressionDatabase/'
    for idx, c in enumerate(ascii_uppercase):
        for i in range(0,10):
            path = dirPath + c + str(i).zfill(2) + '.bmp'     
            im = Image.open(path)
            im = np.asarray(im).flatten() 
            data.append(im)                 

        if(idx >= 9):
            break
    data = np.array(data) / 255 
    return data

def RMSE(X, X_r):
    error = 0.0
    for i in range(len(X)):
        for j in range(64 * 64):           
            error = error + (X[i][j]-X_r[i][j])**2
    error = error / (100 * 64 * 64)
    return(np.sqrt(error))


X = loadData()

X_mean = X.mean(axis=0, keepdims=True)
print(X_mean.shape)
X_ctr = X - X_mean
u, s, v = np.linalg.svd(X_ctr, full_matrices=False)
for k in range(len(X)):
    X_reconstruct = []
    for i in range(len(X)):
        c = np.zeros(k)
        for j in range(k):
            c[j] = np.dot(np.transpose(X_ctr[i]), v[j])
        # sigma weighter eigen vectors
        w = np.zeros(len(v[0]))
        for j in range(k):
            w = w + c[j] * v[j]   
        x_head = (X_mean + w)      
        X_reconstruct.append(x_head)
    X_reconstruct = np.array(X_reconstruct).reshape((100, 64 * 64))
    print( k, RMSE(X, X_reconstruct))
       