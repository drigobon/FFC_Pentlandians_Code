########################################################################
# Purpose: 
#   -Identify Categorical Features from Var. Metadata
#   -Create Indicator Columns for Categorical Features (One-Hot) AND Continuous 'Refuse','Don't Know' Missing Codes
#
# Inputs:
#   -Reduced background.csv-like file with missing codes
#   -Variable Type Metadata file (Included here, found online)
#
# Outputs:
#   -df with one-hot-encoded categorical features
#   -features metadata file
#
# Machine: Laptop, Runtime ~10/15 mins


require(data.table)



## Read stage 1 csv, Features List
#Data_All <- read.csv('background.csv',stringsAsFactors = TRUE)
Data_1 <- read.csv('../output/data_cleaned_stage1_withMissingCodes.csv',stringsAsFactors = TRUE)
Features_All <- read.csv('../data/ffc_variable_types.csv')

Features_1<-Features_All[Features_All$variable %in% colnames(Data_1),] #Clean features
cat_vars <- Features_1[Features_1$variable_type!='continuous',][-1,]


## Filter out Ordinal Variables from non-continuous defined features

Features_FIX <- Features_1
Features_FIX$variable_type_2 <- NA

# More than 15 Unique Answers -> continuous
vars <- Features_FIX[which(Features_FIX$unique_values > 15),]$variable
Features_FIX[Features_FIX$variable %in% vars,]$variable_type <- 'continuous'

# Keywords: How & (Is | Many | Often | Much | Long) | Rate | Frequency | Number | # | Level | Highest | Amount | Days/ | Total | Scale | Times
vars <- Features_FIX[(grepl('how ',Features_FIX$label,ignore.case=TRUE) & (grepl(' is',Features_FIX$label,ignore.case=TRUE) | grepl('many',Features_FIX$label,ignore.case=TRUE) | grepl('often',Features_FIX$label,ignore.case=TRUE) | grepl('much',Features_FIX$label,ignore.case=TRUE) | grepl('long',Features_FIX$label,ignore.case=TRUE)))
                    | grepl('rate',Features_FIX$label,ignore.case=TRUE) | grepl('frequency',Features_FIX$label,ignore.case=TRUE) | grepl('number',Features_FIX$label,ignore.case=TRUE) | grepl('level',Features_FIX$label,ignore.case=TRUE) | grepl('#',Features_FIX$label,ignore.case=TRUE) | grepl('highest',Features_FIX$label,ignore.case=TRUE) | grepl('amount',Features_FIX$label,ignore.case=TRUE) | grepl('days/',Features_FIX$label,ignore.case=TRUE) | grepl('total',Features_FIX$label,ignore.case=TRUE) | grepl('scale',Features_FIX$label,ignore.case=TRUE) | grepl('times',Features_FIX$label,ignore.case=TRUE),]$variable
Features_FIX[(Features_FIX$variable %in% vars) & Features_FIX$variable_type!='continuous',]$variable_type_2 <- 'ordinal'
Features_FIX[Features_FIX$variable %in% vars,]$variable_type <- 'continuous'

# Other unknowns are assumed to be categorical
Features_FIX[Features_FIX$variable_type == 'unknown',]$variable_type <- 'categorical'


# EXCEPTIONS, done manually
Features_FINAL <- Features_FIX

#continuous variables to be labeled categorical (manually looked at)
varlist <- c('hv3ovscale','hv4d2','hv5_ovscale','f5f23c','f5f23e','m5i14a5','m5i14b1','m5i14b2','m5i14b3','m5i14b4','m5j6i','f5i14b1','f5i14b2','f5i14b3','f5i14b4',
             'f5k8','f5k9','p5i36','m2f2d4','',
             'hv3d2','p5childgen_wrong', 'hv3e3' , 'hv4k3', 'f2a7c', 'hv3m14', 'm2a8c', 'm2b33', 'm3a7', 'm3b18', 'f3a7', 'm4a7', 'm4c7f', 'f2fb33',
             'f4c7f', 'f4i0d', 'm5f23c', 'm5f23e', 'p5j8', 'f3b27', 'f4b4b1', 'm5b12b', 't5a2b', 'p5i10', 'f5i13p', 'm5i13p', 'f1j7bc',
             'f1j13b', 'm2f2d5', 'f3f2d1', 'f1e4a', 'm2f2d3', 'm2g8a', 'm2k6b', 'f3f2d2', 'f3f2d3', 'f3f2d4', 'm4f2d4', 'm4i14a', 'f4f2d4',
             'f5a5d01', 'hv4e1', 'm2f2d1', 'm3f2d1', 'm3f2d2', 'm3f2d3', 'm3f2d4', 'hv3f1', 'hv3h1a', 'hv3h1b', 'hv4f2a1',
             'hv4f2b1', 'kind_attention_scale', 'm2f2d2', 'f4f2d1', 'm1f7', 'f1f7', 'm4f2d2', 'm4f2d3', 'f4f2d3', 'm4f2d1', 'f4f2d2',
             'm5a5d02', 'm5a5d04', 'm5a5d03')
Features_FINAL[Features_FINAL$variable %in% varlist,]$variable_type <- 'categorical'

#categorical variables to be labeled continuous (manually looked at)
varlist <- c('m5b28a', 'm2a8d1', 'm2k13a', 'm2k13c')
Features_FINAL[Features_FINAL$variable %in% varlist,]$variable_type <- 'categorical'
Features_FINAL[which(Features_FINAL$variable_type=='categorical'),]$variable_type_2 <- NA

Features_FINAL <-Features_FINAL[-1,] #remove challenge_ID row
cat_vars <- Features_FINAL[Features_FINAL$variable_type=='categorical',]#[-1,]


## New Indicator for NA responses
Data_1[is.na(Data_1)]<- -101


## Indicators for Refuse, Don't Know

##Refusal
refuse <- data.table(apply(data.table(Data_1 == -1),2,as.integer))
col_list <- colnames(refuse)[!(colnames(refuse) %in% cat_vars$variable)]
refuse <- refuse[,col_list,with=F] 
colnames(refuse) <- paste(colnames(Data_1[,col_list]), "_refuse", sep = "")
#Keep col's w/nonzero variance
msk <- apply(refuse, 2, var, na.rm = T) > 0
refuse <- refuse[, msk, with = F]
refuse$challengeID <- Data_1$challengeID
Data_1 <- merge(Data_1, refuse, by = 'challengeID', all=TRUE)


## Don't Know
notknown <- data.table(apply(data.table(Data_1 == -2),2,as.integer))
col_list <- colnames(notknown)[!(colnames(notknown) %in% cat_vars$variable)]
notknown <- notknown[,col_list,with=F]
colnames(notknown) <- paste(colnames(Data_1[,col_list]), "_notknown", sep = "")
## Keep col's w/nonzero variance
msk <- apply(notknown, 2, var, na.rm = T) > 0
notknown <- notknown[, msk, with = F]
notknown$challengeID <- Data_1$challengeID
Data_1 <- merge(Data_1, notknown, by = 'challengeID', all=TRUE)



 

## Non-continuous variables to indicated columns

Data_2 <- Data_1

for (name in cat_vars$variable){
  col_list <- c()
  unique_ans <- unique(Data_2[,name])
  #unique_ans <- unique_ans[!is.na(unique_ans)] ##
  #unique_ans <- unique_ans[unique_ans>=0]      ##
  for (ans in unique_ans){
    colname = paste(name,ans,sep ='_')
    Data_2[,colname]<-as.integer(Data_2[,name]==ans)
    col_list[length(col_list)+1]<-colname
  }
  #ind <- rowSums(Data_2[,col_list])== 0        ##
  #msk <- complete.cases(Data_2[col_list])      ## These 5 lines add NA's in all categorical valid responses for negative integers given
  #Data_2[ind & msk,col_list] <- NA             ## 
}
col_ind <- colnames(Data_2) %in% cat_vars$variable
Data_2<-(Data_2[,!col_ind])


write.csv(Data_2, file = '../output/data_cleaned_stage2.csv',row.names = F)
write.csv(Features_FINAL, file = '../output/features_cleaned_stage2.csv',row.names = F)
