#getwd()
install.packages('mlbench')
install.packages('purrr')
install.packages('caret')
install.packages('xgboost')
install.packages('tictoc')
install.packages('dplyr')
library(caret)
library(xgboost)
library(dplyr)
library(tictoc) 
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
#--------------------------------------------------------------------

#XGBoost in R – via caret, with 5-fold CV simple cross-validation 100 samples


# Step 1: Take a random sample of 100 rows
set.seed(123)
sample_df <- dfdata %>% sample_n(100)

# Step 2: Prepare the data
sample_df$outcome <- as.factor(sample_df$outcome)
levels(sample_df$outcome) <- c("Class0", "Class1")


# Step 3: Set up 5-fold cross-validation
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 1,
  classProbs = TRUE
)

# Step 4: Start timing
tic()

# Step 5: Train the model
xgb_model <- train(
  outcome ~ .,
  data = sample_df,
  method = "xgbTree",
  trControl = fitControl,
  verbose = FALSE
)

# Step 6: Stop timing
timing <- toc(quiet = TRUE)

# Step 7: Extract cross-validated accuracy
best_accuracy <- max(xgb_model$results$Accuracy)

# Step 8: Create the results table
results_df <- data.frame(
  Method_used = "XGBoost in R – via caret, with 5-fold CV simple cross-validation",
  Dataset_size = 100,
  Testing_set_predictive_performance = round(best_accuracy, 4),
  Time_taken_sec = round(timing$toc - timing$tic, 2)
)

# Step 9: Print the results
print(results_df)

#-----------------------------------------------------------------------
#XGBoost in R – via caret, with 5-fold CV simple cross-validation 100 samples


# Step 1: Take a random sample of 1000 rows
set.seed(123)
sample_df <- dfdata %>% sample_n(1000)

# Step 2: Prepare the data
sample_df$outcome <- as.factor(sample_df$outcome)
levels(sample_df$outcome) <- c("Class0", "Class1")


# Step 3: Set up 5-fold cross-validation
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 1,
  classProbs = TRUE
)

# Step 4: Start timing
tic()

# Step 5: Train the model
xgb_model <- train(
  outcome ~ .,
  data = sample_df,
  method = "xgbTree",
  trControl = fitControl,
  verbose = FALSE
)

# Step 6: Stop timing
timing <- toc(quiet = TRUE)

# Step 7: Extract cross-validated accuracy
best_accuracy <- max(xgb_model$results$Accuracy)

# Step 8: Create the results table
results_df <- data.frame(
  Method_used = "XGBoost in R – via caret, with 5-fold CV simple cross-validation",
  Dataset_size = 1000,
  Testing_set_predictive_performance = round(best_accuracy, 4),
  Time_taken_sec = round(timing$toc - timing$tic, 2)
)

# Step 9: Print the results
print(results_df)

#----------------------------------------------------------------

#XGBoost in R – via caret, with 5-fold CV simple cross-validation 10000 samples

# Step 1: Take a random sample of 10000 rows
set.seed(123)
sample_df <- dfdata %>% sample_n(10000)

# Step 2: Prepare the data
sample_df$outcome <- as.factor(sample_df$outcome)
levels(sample_df$outcome) <- c("Class0", "Class1")


# Step 3: Set up 5-fold cross-validation
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 1,
  classProbs = TRUE
)

# Step 4: Start timing
tic()

# Step 5: Train the model
xgb_model <- train(
  outcome ~ .,
  data = sample_df,
  method = "xgbTree",
  trControl = fitControl,
  verbose = FALSE
)

# Step 6: Stop timing
timing <- toc(quiet = TRUE)

# Step 7: Extract cross-validated accuracy
best_accuracy <- max(xgb_model$results$Accuracy)

# Step 8: Create the results table
results_df <- data.frame(
  Method_used = "XGBoost in R – via caret, with 5-fold CV simple cross-validation",
  Dataset_size = 10000,
  Testing_set_predictive_performance = round(best_accuracy, 4),
  Time_taken_sec = round(timing$toc - timing$tic, 2)
)

# Step 9: Print the results
print(results_df)

#--------------------------------------------------------------
#XGBoost in R – via caret, with 5-fold CV simple cross-validation 100000 samples

# Step 1: Take a random sample of 100000 rows
set.seed(123)
sample_df <- dfdata %>% sample_n(100000)

# Step 2: Prepare the data
sample_df$outcome <- as.factor(sample_df$outcome)
levels(sample_df$outcome) <- c("Class0", "Class1")


# Step 3: Set up 5-fold cross-validation
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 1,
  classProbs = TRUE
)

# Step 4: Start timing
tic()

# Step 5: Train the model
xgb_model <- train(
  outcome ~ .,
  data = sample_df,
  method = "xgbTree",
  trControl = fitControl,
  verbose = FALSE
)

# Step 6: Stop timing
timing <- toc(quiet = TRUE)

# Step 7: Extract cross-validated accuracy
best_accuracy <- max(xgb_model$results$Accuracy)

# Step 8: Create the results table
results_df <- data.frame(
  Method_used = "XGBoost in R – via caret, with 5-fold CV simple cross-validation",
  Dataset_size = 100000,
  Testing_set_predictive_performance = round(best_accuracy, 4),
  Time_taken_sec = round(timing$toc - timing$tic, 2)
)

# Step 9: Print the results
print(results_df)
#-------------------------------------------------------------
#XGBoost in R – via caret, with 5-fold CV simple cross-validation 1000000 samples

# Step 1: Take a random sample of 1000000 rows
set.seed(123)
sample_df <- dfdata %>% sample_n(1000000)

# Step 2: Prepare the data
sample_df$outcome <- as.factor(sample_df$outcome)
levels(sample_df$outcome) <- c("Class0", "Class1")


# Step 3: Set up 5-fold cross-validation
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 1,
  classProbs = TRUE
)

# Step 4: Start timing
tic()

# Step 5: Train the model
xgb_model <- train(
  outcome ~ .,
  data = sample_df,
  method = "xgbTree",
  trControl = fitControl,
  verbose = FALSE
)

# Step 6: Stop timing
timing <- toc(quiet = TRUE)

# Step 7: Extract cross-validated accuracy
best_accuracy <- max(xgb_model$results$Accuracy)

# Step 8: Create the results table
results_df <- data.frame(
  Method_used = "XGBoost in R – via caret, with 5-fold CV simple cross-validation",
  Dataset_size = 1000000,
  Testing_set_predictive_performance = round(best_accuracy, 4),
  Time_taken_sec = round(timing$toc - timing$tic, 2)
)

# Step 9: Print the results
print(results_df)

#-----------------------------------------------------------


#XGBoost in R – via caret, with 5-fold CV simple cross-validation 10000000 samples

# Step 1: Take a random sample of 10000000 rows
set.seed(123)
sample_df <- dfdata %>% sample_n(10000000)

# Step 2: Prepare the data
sample_df$outcome <- as.factor(sample_df$outcome)
levels(sample_df$outcome) <- c("Class0", "Class1")


# Step 3: Set up 5-fold cross-validation
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 1,
  classProbs = TRUE
)

# Step 4: Start timing
tic()

# Step 5: Train the model
xgb_model <- train(
  outcome ~ .,
  data = sample_df,
  method = "xgbTree",
  trControl = fitControl,
  verbose = FALSE
)

# Step 6: Stop timing
timing <- toc(quiet = TRUE)

# Step 7: Extract cross-validated accuracy
best_accuracy <- max(xgb_model$results$Accuracy)

# Step 8: Create the results table
results_df <- data.frame(
  Method_used = "XGBoost in R – via caret, with 5-fold CV simple cross-validation",
  Dataset_size = 10000000,
  Testing_set_predictive_performance = round(best_accuracy, 4),
  Time_taken_sec = round(timing$toc - timing$tic, 2)
)

# Step 9: Print the results
print(results_df)