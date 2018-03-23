# Purpose: Generate Elastic Net Predictions
# Inputs: imputed dataset with added homelessness indicators, NO feature selection
# Outputs: predictions from best-performing parameter set
# Machine: 64-core cluster, ~5/6 hours


import os
#import Pickle as pickle

import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':


    target_df = pd.read_csv('../data/train.csv')
    target_df = target_df.set_index('challengeID')

    #data_df_pickle_filepath = "background_imputated.pikcle"
    #if os.path.exists(data_df_pickle_filepath):
    #    with open(data_df_pickle_filepath) as fin:
    #        data_df = pickle.load(fin)
    #else:
        #data_df = pd.read_csv('../data/background.csv')
    data_df = pd.read_csv('../output/data_mean_imputed_Homeless_added.csv')
    data_df = data_df.set_index('challengeID')
    for i, column in enumerate(data_df):
        data_df[column] = pd.to_numeric(data_df[column], errors='coerce')
    data_df = data_df.fillna(0) # imputation
    #with open(data_df_pickle_filepath, "w") as fout:
    #    pickle.dump(data_df, fout,
    #                    protocol=2)

    Xall = data_df.as_matrix()

    target_list = [("gpa", "c"),
                   ("grit", "c"),
                   ("materialHardship", "c"),
                   ("eviction", "b"),
                   ("layoff", "b"),
                   ("jobTraining", "b")]

    parameters = {'n_estimators': [100, 1000],
                  'learning_rate': [0.05, 0.02, 0.01],
                  'max_depth': [2, 5],
                  'subsample': [0.4, 0.6, 0.8],
                  'colsample_bytree': [0.4, 0.6, 0.8]}



    predict_dict = {}
    for target, cl_type in target_list:
        cur_target_df = target_df[~ target_df[target].isnull()]
        cur_data_df = data_df.ix[cur_target_df.index]

        X = cur_data_df.as_matrix()
        y = cur_target_df[target].as_matrix()

        if cl_type == "c":
            clf = GridSearchCV(XGBRegressor(),
                               parameters,
                               scoring='neg_mean_squared_error',
                               n_jobs=-1, cv=3, verbose=2)
            clf.fit(X, y)
            pred_all = clf.predict(Xall)

        else:
            y = y.astype('int')
            clf = GridSearchCV(XGBClassifier(),
                               parameters,
                               scoring='roc_auc',
                               n_jobs=-1, cv=3, verbose=2)
            clf.fit(X, y)
            pred_all = clf.predict_proba(Xall)[:, 1]

        predict_dict[target] = pred_all

    pred_df = pd.DataFrame(predict_dict)
    pred_df.index = data_df.index
    pred_df = pred_df.reset_index()
    pred_df.to_csv("../output/final_pred/xgboost_prediction.csv", index=False)
