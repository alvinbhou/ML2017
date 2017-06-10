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
from sklearn.model_selection import train_test_split
from sklearn import cross_validation, metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error


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


        pred_sj = predictors_sj + ['mean_total_cases_by_month']
        pred_iq = predictors_iq + ['mean_total_cases_by_month']            
                    
        alg_sj.fit(new_train_sj[pred_sj], new_train_sj[target])
        predictions_sj = alg_sj.predict(new_test_sj[pred_sj])        
        alg_iq.fit(new_train_iq[pred_iq], new_train_iq[target])
        predictions_iq = alg_iq.predict(new_test_iq[pred_iq])        
        predictions = np.hstack((predictions_sj, predictions_iq))

        # predictions_sj_first_2_cases = predictions_sj[:1]
        # # do rolling
        # predictions_sj = moving_average(predictions_sj)
        # predictions_sj[:1] = predictions_sj_first_2_cases

        # predictions_iq_first_2_cases = predictions_iq[:1]
        # # do rolling
        # predictions_iq = moving_average(predictions_sj)
        # predictions_iq[:1] = predictions_iq_first_2_cases

        # predictions_iq_first_2_cases = predictions_iq.loc[:1, 'total_cases'].values
        # # do rolling
        # predictions_iq.total_cases = predictions_iq.total_cases.rolling(window=3).mean()
        # predictions_iq.loc[:1,'total_cases'] = predictions_iq_first_2_cases

        print(alg_sj.oob_score_, "this is sj oob")
        print(alg_iq.oob_score_, "this is iq oob")
        # if rolling_step > 0:
        #     predictions = moving_average(predictions, n=rolling_step)
        # print(test_sj[target])
        # print( test_iq[target])

        true_values = np.hstack((test_sj[target], test_iq[target]))
        # print(true_values)
        
        error = metrics.mean_absolute_error(predictions, true_values)
        
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
        error = metrics.mean_absolute_error(predictions, test[target])        
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

sj_merged_train, sj_merged_test = add_groupby_feature(sj, test_sj) 
iq_merged_train, iq_merged_test = add_groupby_feature(iq, test_iq) 
sj_value = sj_merged_train['total_cases']
iq_value = iq_merged_train['total_cases']

ignore_feature_list = ['city','ndvi_sw', 'ndvi_se', 'ndvi_we', 'ndvi_ne', 'week_start_date', 'total_cases']
predictors = [feature for feature in df.columns.tolist() if feature not in ignore_feature_list]
predictors.append('mean_total_cases_by_month')
target = 'total_cases'


X_train_sj, X_valid_sj, y_train_sj, y_valid_sj = train_test_split(sj_merged_train[predictors], sj_value, test_size= 0.123, random_state=42)
X_train_iq, X_valid_iq, y_train_iq, y_valid_iq = train_test_split(iq_merged_train[predictors], iq_value, test_size= 0.123, random_state=42)



# RF BUILD WITH OPTIMIZED PARAMETER
  
rf_sj = RandomForestRegressor(max_features=12, 
                            min_samples_split=2, 
                            n_estimators=810,
                            max_depth=22,
                            min_samples_leaf=1,
                            oob_score=True)



rf_sj = rf_sj.fit(X_train_sj, y_train_sj)

sj_predictions = rf_sj.predict(X_valid_sj)
sj_predictions = np.round(sj_predictions)
sj_accuracy = mean_absolute_error(y_valid_sj, sj_predictions)


print( sj_accuracy, "feelsbad!")

   

rf_iq = RandomForestRegressor(max_features=20, 
                        min_samples_split=9, 
                        n_estimators=180,
                        max_depth=6,
                        min_samples_leaf=5,
                        oob_score=True)
rf_iq = rf_iq.fit(X_train_iq, y_train_iq)
iq_predictions = rf_iq.predict(X_valid_iq)
iq_predictions = np.round(iq_predictions)
iq_accuracy = mean_absolute_error(y_valid_iq, iq_predictions)
print( iq_accuracy, "feelsgood!")
                      
# print(sj_accuracy, iq_accuracy, "feelsgood!")

prediction = np.concatenate((sj_predictions, iq_predictions), axis=0)
true_value = np.concatenate((y_valid_sj, y_valid_iq), axis=0)

print(mean_absolute_error(prediction, true_value))

test_predict_sj = rf_sj.predict(sj_merged_test[predictors])
test_predict_sj = np.round(test_predict_sj)
test_predict_iq = rf_iq.predict(iq_merged_test[predictors])
test_predict_iq = np.round(test_predict_iq)
sj_merged_test['total_cases'] = test_predict_sj
iq_merged_test['total_cases'] = test_predict_iq
result = pd.concat([sj_merged_test, iq_merged_test])
# print(result.head())

# first_2_cases = result['total_cases'].loc[:2]
# print(first_2_cases)
# do rolling
first_2_cases = result['total_cases'].iloc[:2 ]
print(first_2_cases)
# print(result.head())
# result.total_cases.ix[0] = np.nan;
# result.total_cases.ix[1] = np.nan;
# print(result.head())
result.total_cases = result.total_cases.rolling(3).mean()
# result.xs(0, copy = False)['total_cases']=0
result.loc[0 , 'total_cases']=4
result.loc[1 , 'total_cases']=5
print(result.head())

# result['total_cases'].loc[:2] = first_2_cases

result.total_cases = result.total_cases.astype(int)
result[['city','year','weekofyear','total_cases']].to_csv( str(curTime) + 'submission.csv', index=False)
# print(result['total_cases'])


# print(test_predict_sj)


# print(sj_gs.best_score_)



# param_grid = {"max_features" : [8,9,10,11,12,13,14,15]}

# sj_gs = GridSearchCV(estimator=rf_sj,
#                         param_grid = param_grid,
#                         scoring='accuracy',
#                         cv = 5,
#                         n_jobs = -1)

# sj_gs = sj_gs.fit(sj_merged_train[predictors], sj_value)
# print(sj_gs.best_score_)
# print(sj_gs.best_params_)
# print(sj_gs.cv_results_)

# iq_gs = GridSearchCV(estimator=rf_iq,
#                         param_grid = param_grid,
#                         scoring='accuracy',
#                         cv = 5,
#                         n_jobs = -1)

# iq_gs = iq_gs.fit(iq_merged_train[predictors], iq_value)
# print(iq_gs.best_score_)
# print(iq_gs.best_params_)
# print(iq_gs.cv_results_)

# _ = general_rolling_cross_validation(sj_merged_train, iq_merged_train, rf_sj, rf_iq, predictors, predictors)
# result_sj = make_prediction(rf_sj, sj, test_sj, predictors, target)
# result_iq =  make_prediction(rf_iq, iq, test_iq, predictors, target)
# result = pd.concat([result_sj, result_iq])

# # generate result. 
# # get the first_2_cases, cuz moving avg. with window=3 will discard first 2 values
# first_2_cases = result.loc[:1, 'total_cases'].values
# # do rolling
# result.total_cases = result.total_cases.rolling(window=3).mean()
# # get bact the first 2 value
# result.loc[:1,'total_cases'] = first_2_cases
# result.total_cases = result.total_cases.astype(int)
# result.to_csv( str(curTime) + 'submission.csv', index=False)