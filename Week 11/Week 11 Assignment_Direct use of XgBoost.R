#getwd()
install.packages('mlbench')
install.packages('purrr')
install.packages('tictoc')
library(tictoc)
library(xgboost)
library(mlbench)
library(purrr)

data("PimaIndiansDiabetes2")
ds <- as.data.frame(na.omit(PimaIndiansDiabetes2))
## fit a logistic regression model to obtain a parametric equation
logmodel <- glm(diabetes ~ .,
                data = ds,
                family = "binomial")
summary(logmodel)

cfs <- coefficients(logmodel) ## extract the coefficients
prednames <- variable.names(ds)[-9] ## fetch the names of predictors in a vector
prednames

sz <- 100000000 ## to be used in sampling
##sample(ds$pregnant, size = sz, replace = T)

dfdata <- map_dfc(prednames,
                  function(nm){ ## function to create a sample-with-replacement for each pred.
                    eval(parse(text = paste0("sample(ds$",nm,
                                             ", size = sz, replace = T)")))
                  }) ## map the sample-generator on to the vector of predictors
## and combine them into a dataframe

names(dfdata) <- prednames
dfdata

class(cfs[2:length(cfs)])

length(cfs)
length(prednames)
## Next, compute the logit values
pvec <- map((1:8),
            function(pnum){
              cfs[pnum+1] * eval(parse(text = paste0("dfdata$",
                                                     prednames[pnum])))
            }) %>% ## create beta[i] * x[i]
  reduce(`+`) + ## sum(beta[i] * x[i])
  cfs[1] ## add the intercept

## exponentiate the logit to obtain probability values of thee outcome variable
dfdata['outcome'] <- ifelse(1/(1 + exp(-(pvec))) > 0.5,
                            1, 0)

#-----------------------------------------------------------
# XGBoost in R – direct use of xgboost() with simple cross-validation if data size 100

# Sampling 100 rows from your bootstrapped dataset
set.seed(123)
sample_df <- dfdata[sample(nrow(dfdata), 100), ]

# Separating features and target
X <- as.matrix(sample_df[, -which(names(sample_df) == "outcome")])
y <- as.numeric(sample_df$outcome)

# Converting to DMatrix
dtrain <- xgb.DMatrix(data = X, label = y)

# Start timer
tic()

# Run XGBoost with simple cross-validation
cv_result <- xgb.cv(
  data = dtrain,
  nfold = 5,
  nrounds = 5,
  max_depth = 2,
  nthread = 2,
  eta = 1,
  objective = "binary:logistic",
  eval_metric = "error", 
  verbose = 1
)

# End timer
timing <- toc(quiet = TRUE)

# Calculate accuracy
accuracy <- 1 - min(cv_result$evaluation_log$test_error_mean)

# Output the result
cat("Method used: XGBoost in R – direct use of xgboost() with 5-fold CV\n")
cat("Dataset size: 100\n")
cat("Testing-set predictive performance (accuracy):", round(accuracy, 4), "\n")
cat("Time taken for the model to be fit (seconds):", round(timing$toc - timing$tic, 2), "\n")


#------------------------------------------------------------------------------
# XGBoost in R – direct use of xgboost() with simple cross-validation if data size 1000

# Sampling 1000 rows from your bootstrapped dataset
set.seed(123)
sample_df <- dfdata[sample(nrow(dfdata), 1000), ]

# Separating features and target
X <- as.matrix(sample_df[, -which(names(sample_df) == "outcome")])
y <- as.numeric(sample_df$outcome)

# Converting to DMatrix
dtrain <- xgb.DMatrix(data = X, label = y)

# Start timer
tic()

# Run XGBoost with simple cross-validation
cv_result <- xgb.cv(
  data = dtrain,
  nfold = 5,
  nrounds = 5,
  max_depth = 2,
  nthread = 2,
  eta = 1,
  objective = "binary:logistic",
  eval_metric = "error", 
  verbose = 1
)

# End timer
timing <- toc(quiet = TRUE)

# Calculate accuracy
accuracy <- 1 - min(cv_result$evaluation_log$test_error_mean)

# Output the result
cat("Method used: XGBoost in R – direct use of xgboost() with 5-fold CV\n")
cat("Dataset size: 1000\n")
cat("Testing-set predictive performance (accuracy):", round(accuracy, 4), "\n")
cat("Time taken for the model to be fit (seconds):", round(timing$toc - timing$tic, 2), "\n")

#-------------------------------------------------------------------
# XGBoost in R – direct use of xgboost() with simple cross-validation if data size 10000

# Sampling 10000 rows from your bootstrapped dataset
set.seed(123)
sample_df <- dfdata[sample(nrow(dfdata), 10000), ]

# Separating features and target
X <- as.matrix(sample_df[, -which(names(sample_df) == "outcome")])
y <- as.numeric(sample_df$outcome)

# Converting to DMatrix
dtrain <- xgb.DMatrix(data = X, label = y)

# Start timer
tic()

# Run XGBoost with simple cross-validation
cv_result <- xgb.cv(
  data = dtrain,
  nfold = 5,
  nrounds = 5,
  max_depth = 2,
  nthread = 2,
  eta = 1,
  objective = "binary:logistic",
  eval_metric = "error", 
  verbose = 1
)

# End timer
timing <- toc(quiet = TRUE)

# Calculating accuracy
accuracy <- 1 - min(cv_result$evaluation_log$test_error_mean)

# Output the result
cat("Method used: XGBoost in R – direct use of xgboost() with 5-fold CV\n")
cat("Dataset size: 10000\n")
cat("Testing-set predictive performance (accuracy):", round(accuracy, 4), "\n")
cat("Time taken for the model to be fit (seconds):", round(timing$toc - timing$tic, 2), "\n")

#-------------------------------------------------------------------------

# XGBoost in R – direct use of xgboost() with simple cross-validation if data size 100000

# Sampling 100000 rows from your bootstrapped dataset
set.seed(123)
sample_df <- dfdata[sample(nrow(dfdata), 100000), ]

# Separating features and target
X <- as.matrix(sample_df[, -which(names(sample_df) == "outcome")])
y <- as.numeric(sample_df$outcome)

# Converting to DMatrix
dtrain <- xgb.DMatrix(data = X, label = y)

# Start timer
tic()

# Run XGBoost with simple cross-validation
cv_result <- xgb.cv(
  data = dtrain,
  nfold = 5,
  nrounds = 5,
  max_depth = 2,
  nthread = 2,
  eta = 1,
  objective = "binary:logistic",
  eval_metric = "error", 
  verbose = 1
)

# End timer
timing <- toc(quiet = TRUE)

# Calculate accuracy
accuracy <- 1 - min(cv_result$evaluation_log$test_error_mean)

# Output the result
cat("Method used: XGBoost in R – direct use of xgboost() with 5-fold CV\n")
cat("Dataset size: 100000\n")
cat("Testing-set predictive performance (accuracy):", round(accuracy, 4), "\n")
cat("Time taken for the model to be fit (seconds):", round(timing$toc - timing$tic, 2), "\n")

#-------------------------------------------------------------------------

# XGBoost in R – direct use of xgboost() with simple cross-validation if data size 1000000

# Sampling 1000000 rows from your bootstrapped dataset
set.seed(123)
sample_df <- dfdata[sample(nrow(dfdata), 1000000), ]

# Separate features and target
X <- as.matrix(sample_df[, -which(names(sample_df) == "outcome")])
y <- as.numeric(sample_df$outcome)

# Converting to DMatrix
dtrain <- xgb.DMatrix(data = X, label = y)

# Start timer
tic()

# Run XGBoost with simple cross-validation
cv_result <- xgb.cv(
  data = dtrain,
  nfold = 5,
  nrounds = 5,
  max_depth = 2,
  nthread = 2,
  eta = 1,
  objective = "binary:logistic",
  eval_metric = "error", 
  verbose = 1
)

# End timer
timing <- toc(quiet = TRUE)

# Calculate accuracy
accuracy <- 1 - min(cv_result$evaluation_log$test_error_mean)

# Output the result
cat("Method used: XGBoost in R – direct use of xgboost() with 5-fold CV\n")
cat("Dataset size: 1000000\n")
cat("Testing-set predictive performance (accuracy):", round(accuracy, 4), "\n")
cat("Time taken for the model to be fit (seconds):", round(timing$toc - timing$tic, 2), "\n")

#--------------------------------------------------------------------------
# XGBoost in R – direct use of xgboost() with simple cross-validation if data size 10000000

# Sampling 10000000 rows from your bootstrapped dataset
set.seed(123)
sample_df <- dfdata[sample(nrow(dfdata), 10000000), ]

# Separating features and target
X <- as.matrix(sample_df[, -which(names(sample_df) == "outcome")])
y <- as.numeric(sample_df$outcome)

# Converting to DMatrix
dtrain <- xgb.DMatrix(data = X, label = y)

# Start timer
tic()

# Run XGBoost with simple cross-validation
cv_result <- xgb.cv(
  data = dtrain,
  nfold = 5,
  nrounds = 5,
  max_depth = 2,
  nthread = 2,
  eta = 1,
  objective = "binary:logistic",
  eval_metric = "error", 
  verbose = 1
)

# End timer
timing <- toc(quiet = TRUE)

# Calculate accuracy
accuracy <- 1 - min(cv_result$evaluation_log$test_error_mean)

# Output the result
cat("Method used: XGBoost in R – direct use of xgboost() with 5-fold CV\n")
cat("Dataset size: 10000000\n")
cat("Testing-set predictive performance (accuracy):", round(accuracy, 4), "\n")
cat("Time taken for the model to be fit (seconds):", round(timing$toc - timing$tic, 2), "\n")








