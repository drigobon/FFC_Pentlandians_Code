#!/bin/bash

## Reproduces entire Submission

## Create Directories (Assumes 'code','data', and 'output' are created)
mkdir -p ../output &&
mkdir -p ../output/final_pred &&
mkdir -p ../output/LASSO_ALL &&
mkdir -p ../output/LASSO &&
mkdir -p ../output/MI &&
mkdir -p ../output/fig &&

## Data Cleaning
jupyter nbconvert --execute Data_Cleaning_Stage_1.ipynb --ExecutePreprocessor.timeout=-1 &&
Rscript Data_Cleaning_Stage_2.R &&
Rscript Data_Cleaning_Stage_3.R &&
jupyter nbconvert --execute Data_Cleaning_Stage_4.ipynb --ExecutePreprocessor.timeout=-1 &&


## Feature Selection
jupyter nbconvert --execute Mutual_Information_Feat_Selection.ipynb --ExecutePreprocessor.timeout=-1 && 


## LASSO RF (Feature Selection + Predictions)
jupyter nbconvert --execute LassoRandomForest.ipynb --ExecutePreprocessor.timeout=-1 &&

## Elastic Net
./elastic_net_lars_lasso_commands.sh && 

## Random Forest
python RandomForest.py &&

## XGBoost
python XGBoost.py > ../output/XGBoost_parameters.txt &&


## Aggregating Predictions
jupyter nbconvert --execute Team_Prediction_Aggregation.ipynb --ExecutePreprocessor.timeout=-1 &&

## Results and Plots, including extra LASSO Selection
jupyter nbconvert --execute LASSO_run_FS_comp.ipynb --ExecutePreprocessor.timeout=-1 &&
jupyter nbconvert --execute FeatureSelectionComparison.ipynb --ExecutePreprocessor.timeout=-1 &&
python agg_fi_wave_who.py &&
jupyter nbconvert --execute ResultsAnalysis.ipynb --ExecutePreprocessor.timeout=-1 



