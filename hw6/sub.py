import pandas as pd
import os,time
import sys,csv
import numpy as np

# from sklearn import dummy, metrics, cross_validation, ensemble
import keras.optimizers as optimizers
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
from keras.models import load_model

start_time = int(time.time())
# Read in the dataset, and do a little preprocessing,
# mostly to set the column datatypes.
users = pd.read_csv('./data/users.csv', sep='::', 
                        engine='python', 
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']).set_index('UserID')
ratings = pd.read_csv('./data/train.csv', engine='python', 
                          sep=',', names=['TrainDataID', 'UserID', 'MovieID', 'Rating']).set_index('TrainDataID')
movies = pd.read_csv('./data/movies.csv', engine='python',
                         sep='::', names=['MovieID', 'Title', 'Genres']).set_index('MovieID')
testData = pd.read_csv('./data/test.csv', engine='python',
                         sep=',', names=['TestDataID', 'UserID', 'MovieID'])
users = users.drop(users.index[[0]])
ratings= ratings.drop(ratings.index[[0]])
movies = movies.drop(movies.index[[0]])
testData = testData.drop(testData.index[[0]])

movies['Genres'] = movies.Genres.str.split('|')

users.Age = users.Age.astype('category')
users.Gender = users.Gender.astype('category')
users.Occupation = users.Occupation.astype('category')
ratings.MovieID = ratings.MovieID.astype('category')
ratings.UserID = ratings.UserID.astype('category')
testData.MovieID = testData.MovieID.astype('category')
testData.UserID = testData.UserID.astype('category')

# Count the movies and users
n_movies = movies.shape[0]
n_users = users.shape[0]

mframes = [ratings.MovieID, testData.MovieID]
mref = pd.concat(mframes)

mres = pd.DataFrame(mref, columns = ['MovieID'] )
mres.MovieID = mres.MovieID.astype('category')

movieid = mres.MovieID.cat.codes.values
trainMovieID = movieid[0:ratings.MovieID.shape[0]]
testMovieID = movieid[ratings.MovieID.shape[0]:]
print(trainMovieID.shape)
print(testMovieID.shape)

uframes = [ratings.UserID, testData.UserID]
uref = pd.concat(uframes)

ures = pd.DataFrame(uref, columns = ['UserID'] )
ures.UserID = ures.UserID.astype('category')

userid = ures.UserID.cat.codes.values
trainUserID = userid[0:ratings.UserID.shape[0]]
testUserID = userid[ratings.UserID.shape[0]:]
print(trainUserID.shape)
print(testUserID.shape)



model_name = './1496233835model.h5'
model = load_model(model_name)


# scores = model.evaluate([a_movieid, a_userid], a_y, verbose=0)
# print(scores)

# scores = model.evaluate([b_movieid, b_userid], b_y, verbose=0)
# print(scores)

result = model.predict([testMovieID, testUserID])
print(result)

with open(str(start_time) + 'result.csv' , "w", newline='') as mFile:
    writer = csv.writer(mFile)
    writer.writerow(["TestDataID","Rating"])
    for i in range(0, len(result)):
        mFile.write(str(i+1) + ",") 
        x = result[i][0]         
        mFile.write(str(x))
        mFile.write("\n")

