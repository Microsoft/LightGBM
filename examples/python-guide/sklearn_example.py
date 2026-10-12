# coding: utf-8
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

print("Loading data...")
# load or create your dataset
regression_example_dir = Path(__file__).absolute().parents[1] / "regression"
df_train = pd.read_csv(str(regression_example_dir / "regression.train"), header=None, sep="\t")
df_test = pd.read_csv(str(regression_example_dir / "regression.test"), header=None, sep="\t")

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

print("Starting training...")
# train
gbm = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=20)
gbm.fit(X_train, y_train, eval_X=(X_test,), eval_y=(y_test,), eval_metric="l1", callbacks=[lgb.early_stopping(5)])

print("Starting predicting...")
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
print(f"The RMSE of prediction is: {rmse_test}")

# feature importances
print(f"Feature importances: {list(gbm.feature_importances_)}")


# self-defined eval metric
# f(y_true: array, y_pred: array) -> metric_name: str, metric_value: float, maximize: bool
# Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred):
    return "RMSLE", np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False


print("Starting training with custom eval function...")
# train
gbm.fit(X_train, y_train, eval_X=(X_test,), eval_y=(y_test,), eval_metric=rmsle, callbacks=[lgb.early_stopping(5)])


# another self-defined eval metric
# f(y_true: array, y_pred: array) -> metric_name: str, metric_value: float, maximize: bool
# Relative Absolute Error (RAE)
def rae(y_true, y_pred):
    return "RAE", np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(np.mean(y_true) - y_true)), False


print("Starting training with multiple custom eval functions...")
# train
gbm.fit(
    X_train, y_train, eval_X=(X_test,), eval_y=(y_test,), eval_metric=[rmsle, rae], callbacks=[lgb.early_stopping(5)]
)

print("Starting predicting...")
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
rmsle_test = rmsle(y_test, y_pred)[1]
rae_test = rae(y_test, y_pred)[1]
print(f"The RMSLE of prediction is: {rmsle_test}")
print(f"The RAE of prediction is: {rae_test}")

# other scikit-learn modules
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {"learning_rate": [0.01, 0.1, 1], "n_estimators": [20, 40]}

gbm = GridSearchCV(estimator, param_grid, cv=3)
gbm.fit(X_train, y_train)

print(f"Best parameters found by grid search are: {gbm.best_params_}")
