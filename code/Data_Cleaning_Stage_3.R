########################################################################
# Purpose: 
#   -Impute Continuous features w/ Mean
#
# Inputs:
#   -Data Matrix with One-Hot-Encoded categorical features, missing codes ONLY in continuous features
#   -Our created feature metadata file specifying continuous vs categorical
#
# Outputs:
#   -df with mean imputation performed on continuous features, no missing values
#
# Machine: Laptop, Runtime ~10 mins


require(data.table)


## Read stage 2 csv, Features List
#Data_All <- read.csv('background.csv',stringsAsFactors = TRUE)
Data_2 <- read.csv('../output/data_cleaned_stage2.csv',stringsAsFactors = TRUE)
Features_2 <- read.csv('../output/features_cleaned_stage2.csv')

neg_val <- c(-1:-11,-14,-15,-101)

# Remove negative codes
for (n in neg_val){
  Data_2[Data_2 == n] <- NA
}


## Impute with Means
Data_Mean <- Data_2
for (i in Features_2[which(Features_2$variable_type=='continuous'),]$variable){
  Data_Mean[,i][which(!is.finite(Data_Mean[,i]))] <- mean(Data_Mean[,i],na.rm = T)
}

write.csv(Data_Mean, file = '../output/data_mean_imputed.csv',row.names = F)