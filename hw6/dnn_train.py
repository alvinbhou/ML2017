import pandas as pd
import os,time
import sys,csv
import matplotlib

import zipfile
import requests
import numpy as np

from sklearn import dummy, metrics, cross_validation, ensemble
import keras.optimizers as optimizers
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras

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

# print(users)
# print(ratings)

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



# Also, make vectors of all the movie ids and user ids. These are
# pandas categorical data, so they range from 1 to n_movies and 1 to n_users, respectively.
# movieid = ratings.MovieID.cat.codes.values

# trainuserid = ratings.UserID.cat.codes.values
# print(trainuserid.shape)

# And finally, set up a y variable with the rating,
# as a one-hot encoded matrix.
#
# note the '- 1' for the rating. That's because ratings
# go from 1 to 5, while the matrix columns go from 0 to 4

# y = np.zeros((ratings.shape[0], 5))
# y[np.arange(ratings.shape[0]), ratings.Rating - 1] = 1
y = np.array(ratings.Rating)

# Dummy classifier! Just see how well stupid can do.
# pred = dummy.DummyClassifier(strategy='prior')
# pred.fit(ratings[['UserID', 'MovieID']], ratings.Rating)

# print(metrics.mean_absolute_error(ratings.Rating, pred.predict(ratings[['UserID', 'MovieID']])))

# Now, the deep learning classifier

# First, we take the movie and vectorize it.
# The embedding layer is normally used for sequences (think, sequences of words)
# so we need to flatten it out.
# The dropout layer is also important in preventing overfitting
# movie input and embedding
movie_input = keras.layers.Input(shape=[1])
movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 48)(movie_input))
movie_vec = keras.layers.Dropout(0.4)(movie_vec)

# users input and embedding
user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1,48)(user_input))
user_vec = keras.layers.Dropout(0.4)(user_vec)

# dnn architecture
input_vecs = keras.layers.concatenate([movie_vec, user_vec])
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(1024, activation='relu')(input_vecs))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(512, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(256, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
result = keras.layers.Dense(1, activation='relu')(nn)

# Finally, we pull out the result!
# result = keras.layers.Dense(5, activation='softmax')(nn)

# And make a model from it that we can actually run.
model = kmodels.Model([movie_input, user_input], result)
model.summary()
opt = optimizers.adam(lr = 0.0005)
model.compile(opt, loss = 'mean_squared_error')

# If we wanted to inspect part of the model, for example, to look
# at the movie vectors, here's how to do it. You don't need to 
# compile these models unless you're going to train them.
final_layer = kmodels.Model([movie_input, user_input], nn)
movie_vec = kmodels.Model(movie_input, movie_vec)

# Split the data into train and test sets...
a_movieid, b_movieid, a_userid, b_userid, a_y, b_y = cross_validation.train_test_split(trainMovieID, trainUserID, y, test_size=0.1)

# And of _course_ we need to make sure we're improving, so we find the MAE before
# training at all.
# metrics.mean_squared_error(np.argmax(b_y, 1)+1, np.argmax(model.predict([b_movieid, b_userid]), 1)+1)

try:
    history = model.fit([a_movieid, a_userid], a_y, 
                         batch_size = 400,
                         epochs = 45, 
                         validation_data=([b_movieid, b_userid], b_y))
    # plot(history.history['loss'])
    # plot(history.history['val_loss'])
except KeyboardInterrupt:
    pass

model.save(str(start_time) + 'model.h5')

scores = model.evaluate([a_movieid, a_userid], a_y, verbose=0)
print(scores)

scores = model.evaluate([b_movieid, b_userid], b_y, verbose=0)
print(scores)

result = model.predict([testMovieID, testUserID])
print(result)

with open(str(start_time) + 'result.csv' , "w", newline='') as mFile:
    writer = csv.writer(mFile)
    writer.writerow(["TestDataID","Rating"])
    for i in range(0, len(result)):
        mFile.write(str(i+1) + ",")        
        mFile.write(str(result[i][0]))
        mFile.write("\n")

# This is the number that matters. It's the held out 
# test set score. Note the + 1, because np.argmax will
# go from 0 to 4, while our ratings go 1 to 5.
# print(metrics.mean_squared_error(
#     np.argmax(b_y, 1)+1, 
#     np.argmax(model.predict([b_movieid, b_userid]), 1)+1))

# # For comparison's sake, here's the score on the training set.
# print(metrics.mean_squared_error(
#     np.argmax(a_y, 1)+1, 
#     np.argmax(model.predict([a_movieid, a_userid]), 1)+1))