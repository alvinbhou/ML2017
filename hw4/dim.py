import numpy as np
from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues
from sklearn.externals import joblib
import time, sys

# load the pre trained linear SVR
# print('load model')
svr = joblib.load('1494690169.1373851model.pkl') 

# npzfile = np.load('large_data.npz')
# X = npzfile['X']
# y = npzfile['y']

# # we already normalize these values in gen.py
# # X /= X.max(axis=0, keepdims=True)

# svr = SVR(C=10)
# svr.fit(X, y)

test_file = sys.argv[1]
ans_file = sys.argv[2]

testdata = np.load(test_file)
test_X = []
for i in range(200):
    data = testdata[str(i)]
    vs = get_eigenvalues(data)
    test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)

# print('write predict')
with open(ans_file, 'w') as f:
    print('SetId,LogDim', file=f)
    for i, d in enumerate(pred_y):
        x = d
        if(np.abs((np.round(d) - d)) <= 0.2):
            x = np.round(d)
        x = np.log(x) 
        print(f'{i},{x}', file=f)