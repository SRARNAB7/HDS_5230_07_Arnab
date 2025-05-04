#getwd()
install.packages('mlbench')
install.packages('purrr')

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


# Set seed for reproducibility
set.seed(123)

# Sample sizes
sizes <- c(1000, 10000, 100000)

# Loop through sizes and save to CSV
for (sz in sizes) {
  sampled_data <- dfdata[sample(nrow(dfdata), sz), ]
  filename <- paste0("synthetic_logistic_data_", sz, ".csv")
  write.csv(sampled_data, file = filename, row.names = FALSE)
  cat("Saved:", filename, "\n")
}



