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
X_ctr = X - X_mean
print(X_ctr.shape)
u, s, v = np.linalg.svd(X_ctr, full_matrices=False)

eigenFace = []

fig = plt.figure(figsize=(14, 8))
for i in range(0,9):
    eigenFace.append(v[i].reshape((64,64))) 
    ax = fig.add_subplot(3, 3, i+1)
    ax.imshow(eigenFace[i],cmap='gray')
    # plt.show()
# fig.show()
fig.suptitle('Top 9 eigenfaces')
fig.savefig('pca1.png') 


fig2 = plt.figure(figsize=(14, 8))
ax = fig2.add_subplot(1, 1, 1)
ax.imshow(X_mean.reshape((64,64)),cmap='gray')
# fig2.show()
fig2.suptitle('Average face')
fig2.savefig('pcaavg.png') 


#  -- pca2 --

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


fig3 = plt.figure(figsize=(16, 10))
for i in range(len(X)):  
    ax = fig3.add_subplot(10, 10, i+1)
    ax.imshow(pFaces[i],cmap='gray')
    # plt.show()
# fig.show()
fig3.suptitle('10-10 eigenfaces')
fig3.savefig('pca2.png') 





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
    print( k+1, RMSE(X, X_reconstruct))

# print(v.shape)
