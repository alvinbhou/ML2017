import numpy as np
from PIL import Image
from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues
import time
from sklearn.externals import joblib


def loadData():
    data = []
    dirPath = 'hand/hand/hand.seq'
   
    for i in range(1,482):
        print(i)
        path = dirPath + str(i) + '.png'     
        im = Image.open(path)   
        im = np.asarray(im)
        image = np.zeros((480,150))
        for j in range(480):
            image[j] = im[j][100:250]
     
        data.append(image)                 

     
    data = np.array(data) / 255
    return data

data = loadData()
print(data.shape)

svr = joblib.load('1494690169.1373851model.pkl') 

# predict

test_X = []


vs = get_eigenvalues(data)
test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)
print(pred_y)

# with open('3.2ans.csv', 'w') as f:
#     print('SetId,LogDim', file=f)
#     for i, d in enumerate(pred_y):
#         x = np.log(d)
#         if(np.abs(round(x) - x) <= 0.2):
#             x = round(x)
#         print(np.log(d))
#         print(x)
#         print(f'{i},{x}', file=f)