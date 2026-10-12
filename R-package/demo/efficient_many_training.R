# Efficient training means training without giving up too much RAM
# In the case of many trainings (like 100+ models), RAM will be eaten very quickly
# Therefore, it is essential to know a strategy to deal with such issue

# More results can be found here: https://github.com/lightgbm-org/LightGBM/issues/879#issuecomment-326656580
# Quote: "@Laurae2 Thanks for nice easily reproducible example (unlike mine).
# With reset=FALSE you get after 500 iterations (not 1000): OS reports 27GB usage, while R gc() reports 1.5GB.
# Just doing reset=TRUE will already improve things: OS reports 4.6GB.
# Doing reset=TRUE and calling gc() in the loop will have OS 1.3GB. Thanks for the latest tip."

# Load library
library(lightgbm)

# Generate fictive data of size 1M x 100
set.seed(11111L)
x_data <- matrix(rnorm(n = 100000000L, mean = 0.0, sd = 100.0), nrow = 1000000L, ncol = 100L)
y_data <- rnorm(n = 1000000L, mean = 0.0, sd = 5.0)

# Create lgb.Dataset for training
data <- lgb.Dataset(x_data, label = y_data)
data$construct()

# Loop through a training of 1000 models, please check your RAM on your task manager
# It MUST remain constant (if not increasing very slightly)
gbm <- list()

for (i in 1L:1000L) {
  print(i)
  gbm[[i]] <- lgb.train(
      params = list(objective = "regression")
      , data = data
      , 1L
      , reset_data = TRUE
  )
  gc(verbose = FALSE)
}
