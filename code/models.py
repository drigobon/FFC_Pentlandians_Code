#!/usr/bin/python

# this file includes the methods that train logistic, linear regression and multinomial
# models and return them

# Packages Used
import collections
import numpy as np
import utils
#from tobit_regression import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.linear_model import MultiTaskElasticNetCV, MultiTaskElasticNet
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import LassoLarsCV, LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Random Seed
np.random.seed(0)


def train_model(train_features,
                train_labels,
                algorithm,
                num_alphas, 
                skip_cross_validation, alpha, l1_ratio, 
                num_jobs):
  ''' If requested model is multi-elastic-net, then labels must have more than 1 column
  (multi labels).
  '''
  if algorithm == 'elastic-net':
    return train_elasticnet(train_features, train_labels,
                            num_alphas, skip_cross_validation, alpha, l1_ratio,
                            num_jobs)
  elif algorithm == 'lasso':
    return train_lasso(train_features, train_labels,
                       num_alphas, skip_cross_validation, alpha,
                       num_jobs)
  elif algorithm == 'lars':
    return train_lars(train_features, train_labels,
                      num_alphas, skip_cross_validation, alpha,
                      num_jobs)
  elif algorithm == 'multi-elastic-net':
    return train_multi_elasticnet(train_features, train_labels,
                                  num_alphas, skip_cross_validation, alpha, l1_ratio,
                                  num_jobs)
  else:
    sys.exit('Bad algorithm {}'.format(algorithm))


def train_elasticnet(train_features, train_labels,
                     num_alphas,
                     skip_cross_validation, alpha, l1_ratio, 
                     num_jobs):
  """
  Performs the cross validation of elastic net model, and returns the trained model with
  best params. Assume features are scaled/normalized
  """

  best_alpha = alpha
  best_l1_ratio = l1_ratio
  max_iter = 10000
  tol = 0.0005
  if not skip_cross_validation:
    # use 5 fold cross validation
    model = ElasticNetCV(l1_ratio = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975,
                                     0.99, 0.999, 0.9999],
                         max_iter = max_iter,
                         cv = 5,
                         n_alphas = num_alphas,
                         n_jobs = num_jobs,
                         normalize = False,
                         tol = tol)
    model.fit(train_features, train_labels)
    best_alpha = model.alpha_
    best_l1_ratio = model.l1_ratio_
    #print("number of iterations were {}".format(model.n_iter_))
  
  model = ElasticNet(alpha = best_alpha,
                     l1_ratio = best_l1_ratio,
                     normalize = False,
                     max_iter = max_iter,
                     tol = tol)
  model.fit(train_features, train_labels)

  return (model, {'alpha': best_alpha, 'l1_ratio': best_l1_ratio})


def train_lasso(train_features, train_labels,
                num_alphas,
                skip_cross_validation, alpha,
                num_jobs):
  """
  Performs the cross validation of lasso model, and returns the trained model with best
  params. Assume features are scaled/normalized
  """
  best_alpha = alpha
  max_iter = 10000
  tol = 0.0005
  if not skip_cross_validation:
    # use 5 fold cross validation
    model = LassoCV(max_iter = max_iter,
                    cv = 5,
                    n_alphas = num_alphas,
                    n_jobs = num_jobs,
                    normalize = False,
                    tol = tol)
    model.fit(train_features, train_labels)
    best_alpha = model.alpha_
    #print("number of iterations were {}".format(model.n_iter_))
  
  model = Lasso(alpha = best_alpha,
                normalize = False,
                max_iter = max_iter,
                tol = tol)
  model.fit(train_features, train_labels)

  return (model, {'alpha': best_alpha})


def train_lars(train_features, train_labels,
               num_alphas,
               skip_cross_validation, alpha,
               num_jobs):
  """
  Performs the cross validation of lars model, and returns the trained model with best
  params. Assume features are scaled/normalized
  """
  best_alpha = alpha
  max_iter = 10000
  if not skip_cross_validation:
    # use 5 fold cross validation
    model = LassoLarsCV(max_iter = max_iter,
                        cv = 5,
                        max_n_alphas = min(num_alphas, 2000),
                        n_jobs = num_jobs,
                        normalize = False)
    model.fit(train_features, train_labels)
    best_alpha = model.alpha_
    #print("number of iterations were {}".format(model.n_iter_))
 
  model = LassoLars(alpha = alpha,
                    normalize = False,
                    max_iter = max_iter)
  model.fit(train_features, train_labels)

  return (model, {'alpha': best_alpha})


def train_linear(train_features,
                 train_labels,
                 weight_type):
  """
  returns trained OLS model
  Assume features are scaled/normalized
  """
  model = LinearRegression(normalize = True)
  weights = None
  if weight_type == 'linear':
    weights = np.abs(train_labels - np.mean(train_labels))
  elif weight_type == 'root':
    weights = np.sqrt(np.abs(train_labels - np.mean(train_labels)))
  elif weight_type == 'square':
    weights = np.power(np.abs(train_labels - np.mean(train_labels)), 2)
  elif weight_type == 'freq' or weight_type == 'freq-square':
    counts = collections.Counter(train_labels)
    weights_dict = dict()
    max_group_size = max(counts.values())
    for (label, count) in counts.iteritems():
      weights_dict[label] = 1.0 * max_group_size / count
    if weight_type == 'freq':
      weights = [weights_dict[label] for label in train_labels]
    else:
      weights = [(weights_dict[label])**2 for label in train_labels]
  
  model.fit(train_features, train_labels, sample_weight = weights)
  
  return model



#def train_tobit(train_features,
#                train_labels,
#                train_cens):
#  """
#  returns trained tobit model
#  Assume features are scaled/normalized
#  train_cens is an array of size train_labels which signifies which data point is left
#  (-1) or right (1) censored.
#  """
#  train_features_df = pd.DataFrame(train_features)
#  train_labels_series = pd.Series(train_labels)
#  train_cens_series = pd.Series(train_cens)
#  model = TobitModel()
#  model.fit(train_features_df, train_labels_series, train_cens_series, False)
#  
#  return model


def train_multi_elasticnet(train_features, train_labels,
                           num_alphas,
                           skip_cross_validation, alpha, l1_ratio, 
                           num_jobs):
  """
  Performs the cross validation of multi elastic net model, and returns the trained model
  with best params. Assume features are scaled/normalized. Assumes train_labels has more
  than one column.
  """

  best_alpha = alpha
  best_l1_ratio = l1_ratio
  max_iter = 10000
  tol = 0.0005
  if not skip_cross_validation:
    # use 5 fold cross validation
    model = MultiTaskElasticNetCV(l1_ratio = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95,
                                              0.975, 0.99, 0.999, 0.9999],
                         max_iter = max_iter,
                         cv = 5,
                         n_alphas = num_alphas,
                         n_jobs = num_jobs,
                         normalize = False,
                         tol = tol)
    model.fit(train_features, train_labels)
    best_alpha = model.alpha_
    best_l1_ratio = model.l1_ratio_
    #print("number of iterations were {}".format(model.n_iter_))
  
  model = MultiTaskElasticNet(alpha = best_alpha,
                              l1_ratio = best_l1_ratio,
                              normalize = False,
                              max_iter = max_iter,
                              tol = tol)
  model.fit(train_features, train_labels)

  return (model, {'alpha': best_alpha, 'l1_ratio': best_l1_ratio})


def train_random_forest(train_features, train_labels,
                        n_estimators,
                        skip_cross_validation,
                        max_features, min_samples_split, min_samples_leaf,
                        num_jobs):
  """
  Performs the cross validation of random forest model, and returns the trained model with
  best params. Assume features are scaled/normalized
  """

  best_max_features = max_features
  best_min_samples_split = min_samples_split
  best_min_samples_leaf = min_samples_leaf
  if not skip_cross_validation:
    tuned_parameters = [{'n_estimators': [n_estimators],
                         'max_features': ['sqrt', 'log2', 0.1, 0.5],
                         'min_samples_split': [10], 'min_samples_leaf': [3]},
                        {'n_estimators': [n_estimators],
                         'max_features': ['sqrt', 'log2', 0.1, 0.5],
                         'min_samples_split': [20], 'min_samples_leaf': [3, 8]},
                        {'n_estimators': [n_estimators],
                         'max_features': ['sqrt', 'log2', 0.1, 0.5],
                         'min_samples_split': [50], 'min_samples_leaf': [10, 15, 20]},
                        {'n_estimators': [n_estimators],
                         'max_features': ['sqrt', 'log2', 0.1, 0.5],
                         'min_samples_split': [100], 'min_samples_leaf': [15, 20, 40]}]
    model = RandomForestRegressor()
    clf = GridSearchCV(estimator=model,
                       param_grid=tuned_parameters,
                       n_jobs=num_jobs,
                       pre_dispatch='n_jobs',
                       cv=4,
                       scoring='neg_mean_squared_error')
    clf.fit(train_features, train_labels)
    params = clf.best_params_
    #print clf.cv_results_
    #print('Best Cross Validation Score (mean, std): ({},{})'.format(
    #      clf.cv_results_['mean_test_score'][clf.best_index_],
    #      clf.cv_results_['std_test_score'][clf.best_index_]))
    best_max_features = params['max_features']
    best_min_samples_split = params['min_samples_split']
    best_min_samples_leaf = params['min_samples_leaf']

  
  model = RandomForestRegressor(n_estimators = n_estimators,
                                criterion = "mse",
                                max_features = best_max_features,
                                min_samples_split = best_min_samples_split,
                                min_samples_leaf = best_min_samples_leaf,
                                n_jobs = num_jobs)
  model.fit(train_features, train_labels)

  return (model, {'max_features': best_max_features,
                  'min_samples_split': best_min_samples_split,
                  'min_samples_leaf': best_min_samples_leaf})



def train_logistic(train_features,
                   train_labels,
                   class_weight,
                   skip_cross_validation,
                   num_jobs,
                   penalty,
                   cost,
                   multi_class):
  """
  returns the trained logistic model with multinomial or ovr functional form. cost is
  ignored if cross validation is requested. penalty is also ignored if mutli_class is
  multinomial as it only works with l2.
  """

  solver = 'liblinear'
  if multi_class == 'ovr':
    solver = 'liblinear'
  elif multi_class == 'multinomial':
    # multinomial only works with lbfgs and l2 loss
    solver = 'lbfgs'
    penalty = 'l2'
  
  max_iter = 1000
  tol = 0.0005
  best_cost = cost
  if not skip_cross_validation:
    # use 5 fold cross validation
    model = LogisticRegressionCV(Cs = (10.0**np.arange(-6,5)).tolist(),
                                 class_weight = class_weight,
                                 cv = 5,
                                 penalty = penalty,
                                 solver = solver,
                                 tol = tol,
                                 max_iter = max_iter,
                                 n_jobs = num_jobs,
                                 refit = True,
                                 multi_class = multi_class)
  else:
    model = LogisticRegression(C = cost,
                               class_weight = class_weight,
                               penalty = penalty,
                               solver = solver,
                               tol = tol,
                               max_iter = max_iter,
                               multi_class = multi_class)

  model.fit(train_features, train_labels)
 
  if not skip_cross_validation:
    best_cost = model.C_

  return (model, {'cost': best_cost})
