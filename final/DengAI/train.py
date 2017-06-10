from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import time

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm


def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col = [0,1,2])
    df['week'] = df.index.get_level_values('weekofyear') 
    df.loc[:, 'week'] = df['week'] 
    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c',
                 'reanalysis_max_air_temp_k',
                 'reanalysis_min_air_temp_k',
                 'week']
    
    # print(df.isnull().any(axis=1))
    # for index, row in df.iterrows(): 
    #     print(row.)

    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
        # df = df.dropna() 
    else:
        df.fillna(method='ffill', inplace=True)
    
    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    return sj, iq

def get_best_model(train, test, flag):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c + " \
                    "week"
              
    
    grid = 10 ** np.arange(-10, -3, 0.01, dtype=np.float64)
    # print(grid)
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)    
        # predictions = partialRound(predictions)   
        # print(predictions)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model


def partialRound(predictions):
    for i in range(len(predictions)):
        if(np.abs(predictions[i] - np.round(predictions[i])) <= 0.5):
            predictions[i] = np.round(predictions[i])
    return predictions
    

start_time = time.time()

# get data
sj_train, iq_train = preprocess_data('data/dengue_features_train.csv',
                                    labels_path="data/dengue_labels_train.csv")
print(sj_train.shape)
print(iq_train.shape)

# split
sj_train_subtrain = sj_train.head(820)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 820)

iq_train_subtrain = iq_train.head(450)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 450)



# get best model
sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest, 1)
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest, 2)


# test 
sj_test, iq_test = preprocess_data('data/dengue_features_test.csv')

sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv("data/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
# for index, row in submission.iterrows(): 

#     if(np.abs(np.round(row['total_cases']) - row['total_cases']) <= 0.2):
#         value = np.round(row['total_cases'])
#     else:
#         value = row['total_cases']    
#     submission.ix[index].total_cases = value

# print(submission.total_cases)

submission.to_csv("data/" + str(int(start_time)) + "result.csv")