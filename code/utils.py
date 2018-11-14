#!/usr/bin/python

# This file include random forest utility functions to be used by the Elastic Net script

# Packages Used
import sys
import itertools
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler, Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Random Seed
np.random.seed(0)


# function returns a custom callable scorer object to be used by grid search scoring. name
# identifies the type of the scorer to create. Possible names are accuracy,
# weighted-precision, macro-precision, weighted-recall, macro-recall, weighted-f1,
# macro-f1. Theses scorers are different from default version in that they are initialized
# with additional and non-default parameters.
def create_scorer(evaluation):
  if evaluation == "accuracy":
    return make_scorer(accuracy_score)
  elif evaluation == "weighted-precision":
    return make_scorer(precision_score, pos_label = None, average = 'weighted')
  elif evaluation == "macro-precision":
    return make_scorer(precision_score, pos_label = None, average = 'macro')
  elif evaluation == "weighted-recall":
    return make_scorer(recall_score, pos_label = None, average = 'weighted')
  elif evaluation == "macro-recall":
    return make_scorer(recall_score, pos_label = None, average = 'macro')
  elif evaluation == "weighted-f1":
    return make_scorer(f1_score, pos_label = None, average = 'weighted')
  elif evaluation == "macro-f1":
    return make_scorer(f1_score, pos_label = None, average = 'macro')
  elif evaluation == "mae":
    return make_scorer(mean_absolute_error, multioutput='uniform_average')
  elif evaluation == "mse":
    return make_scorer(mean_squared_error, multioutput='uniform_average')
  else:
    sys.exit('Invalid scoring function: ' + scoring_function + ' provided')


# the power to scale gpa labels
GPA_POWER = 2.0
def transform_labels(labels, label_column):
  ''' Function for transforming the labels to appropriate distribution 
  label_column is a single string (for one label) or a list of strings denoting the
  label columns to be predicted.
  The order of vars in label_column should match the order used to train the model and
  order output by predict() method of scikit model.
  '''
  transformed_labels = np.copy(labels)
  if isinstance(label_column, list):
    if 'gpa' in label_column:
      gpa_idx = label_column.index('gpa')
      transformed_labels[:,gpa_idx] = np.power(transformed_labels[:,gpa_idx], GPA_POWER)
  elif isinstance(label_column, str):
    if label_column == 'gpa':
      transformed_labels = np.power(transformed_labels, GPA_POWER)
  else:
    sys.exit('Bad instance type for labels {}'.format(type(label_column)))
  return transformed_labels


def predict(features, model, label_column):
  ''' Function that predict the labels and then transforms them back to original scale
  if any is applied
  label_column is a single string (for one label) or a list of strings denoting the
  label columns to be predicted.
  The order of vars in label_column should match the order used to train the model and
  order output by predict() method of scikit model.
  '''
  predicted = model.predict(features)
  #  predicting multiple columns or a single column?
  if isinstance(label_column, list):
    if 'gpa' in label_column:
      gpa_idx = label_column.index('gpa')
      predicted[:,gpa_idx] = (predicted[:,gpa_idx])**(1.0/GPA_POWER)
  elif label_column == 'gpa':
    predicted = predicted**(1.0/GPA_POWER)
  return predicted


def transform_features(train_features, test_features, transformation, features,
                       minmax_min = 0, minmax_max = 10):
  '''
  features specifies which set of features to apply transformation on
  '''
  if transformation is None:
    if test_features is None:
      return train_features
    else:
      return (train_features, test_features)
  
  if transformation == 'minmax':
    scaler = MinMaxScaler(feature_range = (minmax_min, minmax_max), copy = True)
  elif transformation == 'normal':
    scaler = Normalizer(norm = 'l2', copy = True)
  elif transformation == 'standard':
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
  elif transformation:
    sys.exit('Invalid scaler...{}'.format(transformation))
  scaled_train_features = train_features.copy()
  scaled_train_features[features] = scaler.fit_transform(train_features[features])

  if test_features is None:
    return scaled_train_features

  # have to scale test using the learned scaler only on the availabe train data size
  scaled_test_features = test_features.copy()
  scaled_test_features[features] = scaler.transform(test_features[features])
  return (scaled_train_features, scaled_test_features)



def undersample(features, labels, size):
  """ Returns an undersampled version of features.
  
  The returned 2d array contains size random rows from the original features
  without replacement. The idea is we want to keep the undersampled version as diverse
  as possible.
  if size is negative, no undersampling takes place
  """
  if size < 0:
    return (features, labels)

  num_samples = features.shape[0]
  if num_samples < size:
    sys.exit('Not enough samples: ' + str(num_samples))
  
  indices = np.random.choice(features.shape[0], size, replace=False)
  undersampled_features = features[indices, :]
  undersampled_labels = labels[indices] 
  return (undersampled_features, undersampled_labels)


def prepare_data(input_features_filename,
                 input_labels_filename,
                 input_predictions_filename,
                 transformation,
                 label_column,
                 train_size,
                 test_size,
                 add_transformed_vars,
                 balanced_data,
                 add_interaction_vars):
  # get all train/test features
  features = pd.read_csv(input_features_filename, delimiter=',', index_col=0, header=0,
                         low_memory=False)
  labels = pd.read_csv(input_labels_filename, delimiter=',', index_col=0, header=0,
                       low_memory=False)
  predictions = pd.read_csv(input_predictions_filename, delimiter=',', index_col=0, header=0,
                            low_memory=False)
  features = features.dropna(axis=1, how='all')

  # fit Imputer on all the data for replace missing values
  imputer = Imputer(missing_values="NaN", strategy='mean', axis=0, copy=False)
  imputed_features = imputer.fit_transform(features)
  imputed_features = pd.DataFrame(imputer.fit_transform(features))
  imputed_features.columns = features.columns
  imputed_features.index = features.index
  features = imputed_features
 
  # get list of continuous variables and shift them such that min==0
  # and check if we should add log/square/sqrt transformed var on continuous variables
  feature_unique_vals = features.nunique(axis=0)
  cont_feature_unique_vals = feature_unique_vals[feature_unique_vals > 5]
  cont_feature_names = cont_feature_unique_vals.index.values
  for feature_name in cont_feature_names:
    features[feature_name] = features[feature_name] - features[feature_name].min()

  if add_transformed_vars:
    origin_vars = features.shape[1]
    for feature_name in cont_feature_names:
      # add logof(plus-one) version to features
      features[feature_name + '_log'] = np.log1p(features[feature_name])
      features[feature_name + '_square'] = np.square(features[feature_name])
      features[feature_name + '_sqrt'] = np.sqrt(features[feature_name])
    new_vars = features.shape[1]
    #print 'Added {} new transformed variables.'.format(new_vars - origin_vars)
    cont_feature_names = np.concatenate([cont_feature_names,
                                         cont_feature_names + '_log',
                                         cont_feature_names + '_square',
                                         cont_feature_names + '_sqrt'])
  
  # now scale/transform all continuous features
  transformed_features = pd.DataFrame(transform_features(features, None, transformation,
                                                         cont_feature_names))
  assert transformed_features.shape == features.shape
  transformed_features.columns = features.columns
  transformed_features.index = features.index
  features = transformed_features
  
  # add interaction vars?
  if add_interaction_vars:
    origin_vars = features.shape[1]
    for (feature1, feature2) in itertools.product(features.columns.values,
                                                  features.columns.values):
      interaction_feature = '{}*{}'.format(feature1, feature2)
      features[interaction_feature] = features[feature1] * features[feature2]
    new_vars = features.shape[1]
    #print 'Added {} new interaction variables.'.format(new_vars - origin_vars)

  all_features_predictions_df = features.join(predictions, how='inner')
  assert all_features_predictions_df.shape[0] == features.shape[0]
  feature_names = features.columns.values
  label_names = labels.columns.values
  #print 'Shape of ALL data points df {}'.format(features.shape)
  #print 'Shape of ALL labels df {}'.format(labels.shape)
  #print 'Shape of ALL predictions df {}'.format(predictions.shape)
  #print 'Shape of ALL data points joined with predictions df {}'.format(all_features_predictions_df.shape)
  #print 'Number of ALL features columns in data {}'.format(len(feature_names))
  #print 'Number of ALL label columns in data {}'.format(len(label_names))

  # now only focus on the requested label and remove labels that are nan, since they have
  # no training value. get those rows with known labels
  labels = labels[label_column]
  labels = labels.dropna()
  features_labels = features.join(labels, how='inner')
  assert features_labels.shape[0] == labels.shape[0]
  features = features_labels[feature_names].values
  labels = features_labels[label_column].values
  assert features.shape[0] == labels.shape[0]
  #print 'Number of data points with some available label {}'.format(features.shape[0])
  
  train_features, test_features, train_labels, test_labels = (
      model_selection.train_test_split(features, labels, test_size=test_size))
  
  train_features, train_labels = prepare_train_data(train_features,
                                                    train_labels, balanced_data,
                                                    train_size)
 
  transformed_train_labels = transform_labels(train_labels, label_column)
  transformed_test_labels = transform_labels(test_labels, label_column)
  
  # all features includes the features for all data: train, test and rows with unknown
  # labels. we need to create prediction.csv for these
  return (all_features_predictions_df,
          train_features, transformed_train_labels, train_labels,
          test_features, transformed_test_labels, test_labels,
          feature_names, label_names)


def prepare_train_data(train_features, train_labels, balanced_data, train_size):
  ''' Balances the data set in two possible ways. Scales it to requested size
  '''
  if balanced_data:
    train_features, train_labels = balance_data(train_features, train_labels, train_size)
  else:
    # create requested train size.
    train_features, train_labels = undersample(train_features, train_labels, train_size)
  return (train_features, train_labels)
  


def undersample(features, labels, size):
  """ Returns an undersampled version of features.
  
  The returned 2d array contains size random rows from the original features
  without replacement. The idea is we want to keep the undersampled version as diverse
  as possible.
  if size is negative, no undersampling takes place
  """
  if size < 0:
    return (features, labels)

  num_samples = features.shape[0]
  if num_samples < size:
    sys.exit('Not enough samples: ' + str(num_samples))
  
  indices = np.random.choice(features.shape[0], size, replace=False)
  undersampled_features = features[indices, :]
  undersampled_labels = labels[indices] 
  return (undersampled_features, undersampled_labels)


def balance_data(features, labels, class_values, size=-1):
  """ transform the features and labels such that we have equal number of rows from each 
  class.
  
  size is the requested size per class
  """
  class_values = list(set(labels))
  class_features = dict()
  class_labels = dict()
  class_sizes = dict()
  for val in class_values:
    class_indices = (labels[:] == val)
    class_features[val] = features[class_indices]
    class_labels[val] = labels[class_indices]
    class_sizes[val] = class_features[val].shape[0]
    
  
  if size < 0:
    size_per_class = np.min(class_sizes.values())
  else:
    size_per_class = size/len(class_values)
 
  if np.min(class_sizes.values()) < size_per_class:
    #print('Not enough samples in a class {} vs required class size {}'.
    #      format(class_sizes, size_per_class))
    return (None, None)

  # now undersample each class 
  balanced_features = list()
  balanced_labels = list()
  for val in class_values:
    f, l = undersample(class_features[val], class_labels[val], size_per_class)
    balanced_features.append(f)
    balanced_labels.append(l)

  # concatenate both features and labels
  balanced_features = np.concatenate(balanced_features)
  balanced_labels = np.concatenate(balanced_labels)
  return (balanced_features, balanced_labels)
