# -*- coding: utf-8 -*-
import re
import time
import datetime
import operator
import numpy as np
import pandas as pd
import collections
import unicodedata
import seaborn as sns
import collections
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
 
from sklearn import cross_validation, metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
 
 
from collections import Counter
from datetime import datetime, date, timedelta
 
def make_prediction(alg, train_df, test_df, predictors, target):
 
    alg.fit(train_df[predictors], train_df[target])
    predictions = alg.predict(test_df[predictors])
   
    result_df = test_df[['city', 'year', 'weekofyear']].copy()
    result_df['total_cases'] = predictions
    result_df.total_cases = result_df.total_cases.round()
    result_df.total_cases = result_df.total_cases.astype(int)
   
    return result_df
 
def processing_function(df):
   
    df.fillna(method='ffill', inplace=True)
   
    df_sj = df.loc[df.city == 'sj']
    df_iq = df.loc[df.city == 'iq']
   
    return df_sj, df_iq
 
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.hstack((ret[:n-1],ret[n - 1:] / n))
 
def add_groupby_feature(train_df, test_df):
     
    five_year_ago = train_df.year.max() - 5
    df_month = train_df[train_df.year >= five_year_ago].groupby('month', as_index = False)
 
    feature_name = 'mean_total_cases_by_month'
   
    total_cases_by_month = df_month['total_cases'].mean()
    total_cases_by_month.rename(columns={'total_cases': feature_name}, inplace=True)
 
    train_merged_df = pd.merge(train_df, total_cases_by_month, how = 'left', on='month')
    test_merged_df = pd.merge(test_df, total_cases_by_month, how = 'left', on='month')
   
    return train_merged_df, test_merged_df
 
def general_rolling_cross_validation(df_sj, df_iq, alg_sj, alg_iq, predictors_sj, predictors_iq, target='total_cases',
                                     time_step=4, rolling_step=3):
   
    fold_size_sj = df_sj.shape[0]/(time_step+1)
    fold_size_iq = df_iq.shape[0]/(time_step+1)    
    error_list = []  
   
    for i in range(1, time_step-1):        
        train_sj = df_sj.iloc[: (i+1)*int(fold_size_sj)]
        test_sj = df_sj.iloc[(i+1)*int(fold_size_sj): ]#(i+2)*fold_size_sj]
       
        train_iq = df_iq.iloc[: (i+1)*int(fold_size_iq)]
        test_iq = df_iq.iloc[(i+1)*int(fold_size_iq): ]#(i+2)*fold_size_iq]
           
        new_train_sj, new_test_sj = add_groupby_feature(train_sj, test_sj)
        new_train_iq, new_test_iq = add_groupby_feature(train_iq, test_iq)
#         print(new_train_sj)
#         print(new_train_iq)
 
        pred_sj = predictors_sj + ['mean_total_cases_by_month']
        pred_iq = predictors_iq + ['mean_total_cases_by_month']            
                   
        alg_sj.fit(new_train_sj[pred_sj], new_train_sj[target])
        predictions_sj = alg_sj.predict(new_test_sj[pred_sj])        
        alg_iq.fit(new_train_iq[pred_iq], new_train_iq[target])
        predictions_iq = alg_iq.predict(new_test_iq[pred_iq])        
        predictions = np.hstack((predictions_sj, predictions_iq))
       
        if rolling_step > 0:
            predictions = moving_average(predictions, n=rolling_step)
        # print(test_sj[target])
        # print( test_iq[target])
 
        true_values = np.hstack((test_sj[target], test_iq[target]))
        # print(true_values)
       
        error = metrics.mean_absolute_error(np.round(predictions), true_values)
       
        error_list.append(error)
       
   
    print ("Rolling cross-validation: ")
    print ("Mean {0:.2f} - Median {1:.2f} - Std {2:.2f}".format(np.mean(error_list), np.median(error_list), np.std(error_list)))  
    return error_list
       
 
# seems to be deprecated
def rolling_cross_validation(df, alg, predictors, target='total_cases', time_step=4, rolling_step=3):
   
    fold_size = df.shape[0]/(time_step+1)    
    error_list = []
   
    for i in range(1,time_step-1):
       
        train = df.iloc[: (i+1)*fold_size]
        test = df.iloc[(i+1)*fold_size: (i+2)*fold_size]
           
        new_train, new_test = add_groupby_feature(train, test)
        pred = predictors + ['mean_total_cases_by_month']
 
        alg.fit(new_train[pred], new_train[target])
        predictions = alg.predict(new_test[pred])        
     
        if rolling_step > 0:
            predictions = moving_average(predictions, n=rolling_step)        
        error = metrics.mean_absolute_error(np.round(predictions), test[target])        
        error_list.append(error)
       
   
    #print "Rolling cross-validation: ",
    #print "Mean {0:.2f} - Median {1:.2f} - Std {2:.2f}".format(np.mean(error_list), np.median(error_list), np.std(error_list))
   
    return np.mean(error_list)
 
 
curTime = int(time.time())
 

# load data
feature = pd.read_csv('data/dengue_features_train_m.csv', infer_datetime_format=True)
label = pd.read_csv('data/dengue_labels_train.csv')
test = pd.read_csv('data/dengue_features_test_m.csv')
df = pd.merge(feature, label, how='outer', on=label.columns.tolist()[:-1])
df = df.dropna(axis=0, thresh=20)
df_sj = df[df.city == 'sj'].copy()
df_iq = df[df.city == 'iq'].copy()
 
sj, iq = processing_function(df)
test_sj, test_iq = processing_function(test)
 
ignore_feature_list = ['city', 'ndvi_ne', 'week_start_date', 'total_cases']
predictors = [feature for feature in df.columns.tolist() if feature not in ignore_feature_list]
target = 'total_cases'
 
# RF BUILD WITH OPTIMIZED PARAMETER
rf_sj = RandomForestRegressor(max_features=10,
                              min_samples_split=88,
                              n_estimators=674,
                              max_depth=18,
                              min_samples_leaf=3)
 
rf_iq = RandomForestRegressor(max_features=10,
                              min_samples_split=19,
                              n_estimators=200,
                              max_depth=16,
                              min_samples_leaf=5)
                             
_ = general_rolling_cross_validation(sj, iq, rf_sj, rf_iq, predictors, predictors)
result_sj = make_prediction(rf_sj, sj, test_sj, predictors, target)
result_iq =  make_prediction(rf_iq, iq, test_iq, predictors, target)
result = pd.concat([result_sj, result_iq])
 
# generate result.
# get the first_2_cases, cuz moving avg. with window=3 will discard first 2 values
print(result.total_cases)
first_2_cases = result.loc[:1, 'total_cases'].values
print(first_2_cases)
# do rolling
result.total_cases = result.total_cases.rolling(window=3).mean()
print(result.total_cases)
# get bact the first 2 value
result.loc[:1,'total_cases'] = first_2_cases
result.total_cases = result.total_cases.astype(int)
result.to_csv( str(curTime) + 'submission.csv', index=False)