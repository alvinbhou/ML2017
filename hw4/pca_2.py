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
    return np.array(data)


X = loadData()

X_mean = X.mean(axis=0, keepdims=True)
X_ctr = X - X_mean
u, s, v = np.linalg.svd(X_ctr, full_matrices=False)


top5eigen= v[0:5]
pFaces = []


# get weight
for i in range(len(X)):
    c = np.zeros(5)
    for j in range(5):
        c[j] = np.dot(np.transpose(X_ctr[i]), v[j])
    # sigma weighter eigen vectors
    w = np.zeros(len(v[0]))
    for j in range(5):
        w = w + c[j] * v[j]   
    x_head = (X_mean + w).reshape((64,64))     
    pFaces.append(x_head)
    # print(x_head.shape)

 


fig = plt.figure(figsize=(16, 10))
for i in range(len(X)):  
    ax = fig.add_subplot(10, 10, i+1)
    ax.imshow(pFaces[i],cmap='gray')
    # plt.show()
fig.show()
fig.suptitle('10-10 eigenfaces')
fig.savefig('pca2.png') 



fig2 = plt.figure(figsize=(16, 10))
for i in range(len(X)):  
    ax = fig2.add_subplot(10, 10, i+1)
    ax.imshow(X[i].reshape((64,64)) ,cmap='gray')
    # plt.show()
fig2.show()
fig2.suptitle('Original faces')
fig2.savefig('pca2origin.png') 


# print(len(top5eigen))