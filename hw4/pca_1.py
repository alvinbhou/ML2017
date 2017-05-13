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
fig.show()
fig.suptitle('Top 9 eigenfaces')
fig.savefig('pca1norm.png') 
# print(v.shape)
