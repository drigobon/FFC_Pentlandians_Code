TOP_LEVEL_DIR="../"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/code/"
RAW_DATA_DIR="${TOP_LEVEL_DIR}/data/"
FEATURE_SELECTED_DATA_DIR="${TOP_LEVEL_DIR}/output/MI"
FEATURES_INPUT_FILE="${FEATURE_SELECTED_DATA_DIR}/data_univariate_feature_selection_100.csv"
LABELS_INPUT_FILE="${RAW_DATA_DIR}/train.csv"
PREDICTIONS_INPUT_FILE="${RAW_DATA_DIR}/prediction.csv"

RESULTS_DIR="${TOP_LEVEL_DIR}/output/final_pred/"
PREDICTIONS_PREFIX="${RESULTS_DIR}/elastic_prediction"
PARAMETERS_PREFIX="${TOP_LEVEL_DIR}/output/elastic_parameters.txt"

# number of cores to use in grid search
NUM_JOBS=1

LABEL_COLUMNS="gpa grit materialHardship"

for ALGORITHM in "elastic-net"
do
  for TRANSFORMATION in "normal" # "standard" 
  do
    python ${SCRIPTS_DIR}/elastic_net_lars_lasso.py \
      -al $ALGORITHM -t $TRANSFORMATION -tes 100 -lc $LABEL_COLUMNS -nj $NUM_JOBS \
      $FEATURES_INPUT_FILE \
      $LABELS_INPUT_FILE \
      $PREDICTIONS_INPUT_FILE \
      ${PREDICTIONS_PREFIX}.csv > \
      ${PARAMETERS_PREFIX}
  done 
done
