# Purpose: Generate Elastic Net Predictions
# Inputs: outputs of MI Feature Selection
# Outputs: Prediction file
# Machine: Laptop, ~10/15 mins


import argparse
import sys
import numpy as np
import utils
import models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

parser = argparse.ArgumentParser(
  description='Elastic net script.')
parser.add_argument('input_features_filename',
                    help='input file containing the all the features per child.')
parser.add_argument('input_labels_filename',
                    help='input file containing the all the known labels per child. '
                    'It must be joinable with input_features_filename by the challenge '
                    'id column.')
parser.add_argument('input_predictions_filename',
                    help='input file containing the features we need to make prediction '
                    'for their unknown labels.')
parser.add_argument('output_prediction_filename',
                    help='The output file will contain our predictions for the rows in '
                    'input_prediction_filename.')
parser.add_argument('-al', '--algorithm', dest='algorithm',
                    default='elastic-net', choices=['elastic-net', 'lasso', 'lars'],
                    help='The algorithm to use for training/prediction.')
parser.add_argument('-t', '--transformation', dest='transformation',
                    default='standard', choices=['standard', 'normal', 'minmax'],
                    help='The method to transform features: normalize, standard scaling '
                    'or minmax scaling.')
parser.add_argument('-trs', '--train_size', dest='train_size',
                    type=int, default=-1,
                    help='The size of train set. It must be a whole integer, greater '
                    'than 1 which represents the absolute number of data points to use '
                    'for training. Default is -1 which means all available data after '
                    'splitting into test and train is used. There will be some '
                    'undersampling of the majority class, if train data needs to be '
                    'balanced.')
parser.add_argument('-tes', '--test_size', dest='test_size',
                    type=int, default=800,
                    help='The size of test set. It must be a whole integer, greater '
                    'than 1 which represents the absolute number of data points to use '
                    'for testing. You need to make sure test and train size do not '
                    'exceed the total size and there are enough samples of each class '
                    'in case train data is balanced.')
parser.add_argument('-lc', '--label_columns', dest='label_columns',
                    nargs='+', default=['gpa', 'grit', 'materialHardship'],
                    help='The column names in input_labels_filename to predict. '
                    'We essentially build a model for each requested column.')
parser.add_argument('-nj', '--num_jobs', dest='num_jobs', type=int, default=-1,
                    help='Number of jobs to instantiate for grid search. Default is '
                    '-1, which corresponds to the number of cores in the machine')

# cross validation flags: they will be ignored if cross validation is not skipped.
parser.add_argument('-sc', '--skip_cross_validation', dest='skip_cross_validation',
                    default=False, action='store_true',
                    help='If specified, skips cross validation and uses the parameters '
                    'given below. ')
parser.add_argument('-na', '--num_alphas', dest='num_alphas', type=int, default=200,
                    help='The number of alphas to try for each l1 ratio, in case '
                    'cross validation is not skipped. Defaults to 100.')
parser.add_argument('-l', '--l1_ratio', dest='l1_ratio', default=0.15, type=float,
                    help='The Elastic Net mixing parameter, must be a float between 0 '
                    'and 1, with 0 <= l1_ratio <= 1. It determines how aggressive is the '
                    'feature selection in elastic net. Must be specified only if penatly '
                    'is elasticnet, as default (0.15) might not be good.  Relevent only '
                    'if cross validation is skipped.')
parser.add_argument('-a', '--alpha', dest='alpha', default=1, type=float,
                    help='alpha multiplies the penalty terms and thus determines '
                    'the level of regularization. Defaults to 1.0. alpha = 0 is '
                    'equivalent to an ordinary least square.')
args = parser.parse_args()

def main():
  add_transformed_vars = True
  
  all_predictions = None
  for label_column in args.label_columns:
    #print 'Training for Label {}'.format(label_column)
    (all_features_predictions_df,
     train_features, transformed_train_labels, train_labels,
     test_features, transformed_test_labels, test_labels,
     all_feature_names, all_label_names) = utils.prepare_data(args.input_features_filename,
                                                              args.input_labels_filename,
                                                              args.input_predictions_filename,
                                                              args.transformation,
                                                              label_column,
                                                              args.train_size,
                                                              args.test_size,
                                                              add_transformed_vars,
                                                              False,
                                                              False)
    all_features = np.concatenate((train_features, test_features), axis=0)
    all_labels = np.concatenate((train_labels, test_labels), axis=0)
    all_transformed_labels = np.concatenate((transformed_train_labels, transformed_test_labels), axis=0)
    #print 'Max label values: {}'.format(np.max(all_labels))
    #print 'Min label values: {}'.format(np.min(all_labels))
    if all_predictions is None:
      all_predictions = all_features_predictions_df[all_label_names]
    
    #print('\n*****************************\n')
    (model, best_params) = models.train_model(all_features,
                                              all_transformed_labels,
                                              args.algorithm,
                                              args.num_alphas, 
                                              args.skip_cross_validation,
                                              args.alpha, args.l1_ratio, 
                                              args.num_jobs)
    train_y_true = all_labels
    train_y_pred = utils.predict(all_features, model, label_column)
    #print 'R-Squared on train is {}'.format(r2_score(train_y_true, train_y_pred))
    
    #print('\n')
    #print 'MAE on train: {}'.format(mean_absolute_error(train_y_true, train_y_pred))
    #print 'MSE on train: {}'.format(mean_squared_error(train_y_true, train_y_pred))
    
    coefs = model.coef_
    #print 'Optimal parameters in final model is {}'.format(best_params)
    # info on most predictive coefs
    #print 'Model has {} non-zero coefficients.'.format(len(coefs[coefs != 0]))
    highest_indices = np.abs(coefs).argsort()[::-1]
    #print 'Top predictors of {}'.format(label_column)
    for i in range(min(10, len(coefs[coefs != 0]))):
      idx = highest_indices[i]
    #  print '{}: {}'.format(all_feature_names[idx], coefs[idx])

    # predict on all and write to file
    prediction_features = all_features_predictions_df[all_feature_names]
    label_predictions = utils.predict(prediction_features, model, label_column)
    # correct if below/above min/max
    label_predictions[label_predictions < np.min(all_labels)] = np.min(all_labels)
    label_predictions[label_predictions > np.max(all_labels)] = np.max(all_labels)
    all_predictions.loc[:,label_column] = label_predictions
    
    #print('\n*****************************')
    #print('*****************************\n')
  all_predictions.to_csv(args.output_prediction_filename)


if __name__ == '__main__':
  main()
