#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Purpose: Create Random Forest Predictions using 3-fold Cross-Validation to determine best model predictions and best model parameters
# Inputs: Feature Selection file from MI, K value specified in string (100 used for best predictions)
# Outputs: Prediction files for best RF, average best RF, and weighted best RF by CV score
# Machine: High-performance Cluster (64 cores), ~4 hrs


# In[1]:


# Packages Used
import pandas as pd
import numpy as np
import scipy.stats as sp
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import brier_score_loss, mean_squared_error
from scipy.stats import randint, uniform ### IMPORTANT: these are distributions and not draws
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics.scorer import make_scorer
import copy


# In[2]:


# Random Seed
np.random.seed(0)


# In[3]:


##loading the data
dfx = pd.read_csv('../output/MI/data_univariate_feature_selection_100.csv',index_col='challengeID')
dfy = pd.read_csv('../data/train.csv',index_col='challengeID')
  
predictions = {'challengeID':np.array(list(dfx.index)),
               'gpa':None,'grit':None,'materialHardship':None,'eviction':None,'layoff':None,'jobTraining':None}   


# In[4]:


##scaling X
for col in dfx.columns:
    dfx[col] = (dfx[col] - dfx[col].mean())/dfx[col].std(ddof=0)


# In[5]:


outcomes = list(dfy.columns) #get the names of the outcomes


# In[6]:


np.shape(dfx)


# In[7]:


randint(1,300)


# In[11]:


NUM_MODELS = 50
n_CVjobs = 10
n_CVsplits = 5
n_modelJobs = 10 #*4 ##remove the comment on EC2
mode = None
n_iter_search = 50
max_features = 150 ##this should be < n_features

reg_outcomes = ['gpa', 'grit', 'materialHardship']
clf_outcomes = [ 'eviction', 'layoff', 'jobTraining']




__reg_param_dist = {'max_depth': [1,2,3,4,None],
                    'max_features': randint(1, max_features),
                    'min_samples_split':randint(2, 300),
                    'min_samples_leaf':randint(1, 300),
                    'n_estimators':randint(50, 500),
                    'oob_score':[True,False]}

__clf_param_dist = {'max_depth': [1,2,3,4,None],
                    'max_features': randint(1, max_features),
                    'min_samples_split':randint(2, 300),
                    'min_samples_leaf':randint(1, 300),
                    'n_estimators':randint(50, 500),
                    'criterion':['gini','entropy']}


###### We don't use this anymore (where we average the parameters of the model)####
#__reg_param = {'max_depth': [],
#               'max_features': [],
#               'min_samples_split':[],
#               'min_samples_leaf':[],
#               'n_estimators':[],
#               'oob_score':[]}
#
#__clf_param = {'max_depth': [],
#               'max_features': [],
#               'min_samples_split':[],
#               'min_samples_leaf':[],
#               'n_estimators':[],
#               'criterion':[]}
#best_param = {'reg' : __reg_param,
#              'clf': __clf_param}

param_dist = {'reg' : __reg_param_dist,
              'clf': __clf_param_dist}

model = {'reg' : RandomForestRegressor(n_jobs=n_modelJobs),
          'clf': RandomForestClassifier(n_jobs=n_modelJobs)}

scorer = {'reg' : make_scorer(mean_squared_error,greater_is_better=False),
           'clf' : make_scorer(brier_score_loss,greater_is_better=False)}

evaluate_error = {'reg': mean_squared_error,
                  'clf': brier_score_loss}


best_model_prediction = {'challengeID':np.array(list(dfx.index)),
               'gpa':None,
               'grit':None,
               'materialHardship':None,
               'eviction':None,
               'layoff': None,
               'jobTraining':None
              }

avg_models_prediction = copy.deepcopy(best_model_prediction)
weighted_models_prediction = copy.deepcopy(best_model_prediction)


# In[ ]:


for outcome in outcomes:
    ##Figure out in what mode we are
    if outcome in reg_outcomes:
        mode = 'reg'
    else:
        mode = 'clf'
    
    ###prepare X and Y####
    full = dfx.join(dfy, how='outer') #connect the background data to outcomes
    full_X = full.copy()
    for inner_outcome in outcomes:
        del full[inner_outcome]
    X = full_X.dropna(subset=[outcome], how='all')
    y = X[outcome]
    for inner_outcome in outcomes:
        del full_X[inner_outcome]

    for inner_outcome in outcomes:
        del X[inner_outcome]
        
    ##In order to try the different aggregation mechanisms
    combined_model_prediction = {'challengeID':np.array(list(dfx.index)),outcome: None}
    lowest_error = np.inf
    best_model = None
    all_models_scores = []
    weighted_models = [] 
    n_good_models = 0

    for i in range(1,NUM_MODELS+1):
        print('at loop:',i,'for outcome ', outcome)
        ##prepare the nested CV
        inner_cv = StratifiedKFold(n_splits=n_CVsplits, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=n_CVsplits, shuffle=True, random_state=i)

        ########Nested CV with parameter optimization########
        #the Randomized search for the best parameters through cross validation
        search = RandomizedSearchCV(estimator=model[mode], param_distributions=param_dist[mode],
                                    scoring = scorer[mode],random_state=i+1,
                                    cv=inner_cv,n_jobs=n_CVjobs,n_iter=n_iter_search)
        search.fit(X, y)
        #The evaluation of the best model found by the inner CV by having an outer CV
        nested_score = cross_val_score(search, X=X, y=y, cv=outer_cv)
        if mode == 'reg':
            prediction = search.best_estimator_.predict(X)
        else:
            prediction = search.best_estimator_.predict_proba(X)[:,1]

        if evaluate_error[mode](y,prediction) < np.inf:
            n_good_models +=1

            if evaluate_error[mode](y,prediction) < lowest_error:
                print('best so far')
                lowest_error = evaluate_error[mode](y,prediction)
                best_model = search.best_estimator_

            if i == 1:
                if mode == 'reg':
                    combined_model_prediction[outcome] = search.best_estimator_.predict(full_X)
                else:
                    combined_model_prediction[outcome] = search.best_estimator_.predict_proba(full_X)[:,1]
            else:
                if mode == 'reg':
                    combined_model_prediction[outcome] = combined_model_prediction[outcome] + search.best_estimator_.predict(full_X)
                else:
                    combined_model_prediction[outcome] = combined_model_prediction[outcome]+ search.best_estimator_.predict_proba(full_X)[:,1]

                

            all_models_scores.append(evaluate_error[mode](y,prediction))
            if mode == 'reg':
                weighted_models.append(search.best_estimator_.predict(full_X))
            else:
                weighted_models.append(search.best_estimator_.predict_proba(full_X)[:,1])
        print('score:', evaluate_error[mode](y,prediction))
        print('CV scores:', nested_score.mean(), nested_score)
        print('best params', search.best_params_)
        print('#######')
    

        
        ##best model prediction
        best_model.fit(X, y)
        if mode == 'reg':
            final_prediction = best_model.predict(full_X)
        else:
            final_prediction = best_model.predict_proba(full_X)[:,1]
        best_model_prediction[outcome] = final_prediction
        
        ##avg models prediction
        avg_models_prediction[outcome] = combined_model_prediction[outcome]/float(n_good_models)
        
        
        ##weighted models prediction
        scores = np.array(all_models_scores)
        normlized_scores = scores/sum(scores)
        normlized_scores = normlized_scores.reshape(normlized_scores.shape[0],1)
        models_predicitons = np.matrix(weighted_models).T
        weighted_prediction = np.array(models_predicitons*normlized_scores).flatten().tolist()
        weighted_models_prediction[outcome] = weighted_prediction

        


# In[22]:


df_best = pd.DataFrame.from_dict(best_model_prediction)
df_avg = pd.DataFrame.from_dict(avg_models_prediction)
df_weighted = pd.DataFrame.from_dict(weighted_models_prediction)


# In[24]:


#df_best.to_csv('../output/final_pred/best_multiRF_prediction.csv',index=False)
#df_avg.to_csv('../output/final_pred/avg_multiRF_prediction.csv',index=False)
df_weighted.to_csv('../output/final_pred/weighted_multiRF_prediction.csv',index=False)

