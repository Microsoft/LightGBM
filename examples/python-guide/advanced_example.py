# coding: utf-8
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

print("Loading data...")
# load or create your dataset
binary_example_dir = Path(__file__).absolute().parents[1] / "binary_classification"
df_train = pd.read_csv(str(binary_example_dir / "binary.train"), header=None, sep="\t")
df_test = pd.read_csv(str(binary_example_dir / "binary.test"), header=None, sep="\t")
W_train = pd.read_csv(str(binary_example_dir / "binary.train.weight"), header=None)[0]
W_test = pd.read_csv(str(binary_example_dir / "binary.test.weight"), header=None)[0]

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

num_train, num_feature = X_train.shape

# generate feature names
feature_name = [f"feature_{col}" for col in range(num_feature)]

# create dataset for lightgbm
# if you want to re-use data, remember to set free_raw_data=False
lgb_train = lgb.Dataset(
    X_train, y_train, weight=W_train, feature_name=feature_name, categorical_feature=[21], free_raw_data=False
)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, weight=W_test, free_raw_data=False)

# specify your configurations as a dict
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}

print("Starting training...")
# feature_name and categorical_feature
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=10,
    valid_sets=lgb_train,  # eval training data
)

print("Finished first 10 rounds...")
# check feature name
print(f"7th feature name is: {lgb_train.feature_name[6]}")

print("Saving model...")
# save model to file
gbm.save_model("model.txt")

print("Dumping model to JSON...")
# dump model to JSON (and save to file)
model_json = gbm.dump_model()

with open("model.json", "w+") as f:
    json.dump(model_json, f, indent=4)

# feature names
print(f"Feature names: {gbm.feature_name()}")

# feature importances
print(f"Feature importances: {list(gbm.feature_importance())}")

print("Loading model to predict...")
# load model to predict
bst = lgb.Booster(model_file="model.txt")
# can only predict with the best iteration (or the saving iteration)
y_pred = bst.predict(X_test)
# eval with loaded model
auc_loaded_model = roc_auc_score(y_test, y_pred)
print(f"The ROC AUC of loaded model's prediction is: {auc_loaded_model}")

print("Dumping and loading model with pickle...")
# dump model with pickle
with open("model.pkl", "wb") as fout:
    pickle.dump(gbm, fout)
# load model with pickle to predict
with open("model.pkl", "rb") as fin:
    pkl_bst = pickle.load(fin)
# can predict with any iteration when loaded in pickle way
y_pred = pkl_bst.predict(X_test, num_iteration=7)
# eval with loaded model
auc_pickled_model = roc_auc_score(y_test, y_pred)
print(f"The ROC AUC of pickled model's prediction is: {auc_pickled_model}")

# continue training
# init_model accepts:
# 1. model file name
# 2. Booster()
gbm = lgb.train(params, lgb_train, num_boost_round=10, init_model="model.txt", valid_sets=lgb_eval)

print("Finished 10 - 20 rounds with model file...")

# decay learning rates
# reset_parameter callback accepts:
# 1. list with length = num_boost_round
# 2. function(curr_iter)
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=10,
    init_model=gbm,
    valid_sets=lgb_eval,
    callbacks=[lgb.reset_parameter(learning_rate=lambda iter: 0.05 * (0.99**iter))],
)

print("Finished 20 - 30 rounds with decay learning rates...")

# change other parameters during training
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=10,
    init_model=gbm,
    valid_sets=lgb_eval,
    callbacks=[lgb.reset_parameter(bagging_fraction=[0.7] * 5 + [0.6] * 5)],
)

print("Finished 30 - 40 rounds with changing bagging_fraction...")


# self-defined objective function
# f(preds: array, train_data: Dataset) -> grad: array, hess: array
# log likelihood loss
def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


# self-defined eval metric
# f(preds: array, train_data: Dataset) -> metric_name: str, metric_value: float, maximize: bool
# binary error
# NOTE: when you do customized loss function, the default prediction value is margin
# This may make built-in evaluation metric calculate wrong results
# For example, we are doing log likelihood loss, the prediction is score before logistic transformation
# Keep this in mind when you use the customization
def binary_error(preds, train_data):
    labels = train_data.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    return "error", np.mean(labels != (preds > 0.5)), False


# Pass custom objective function through params
params_custom_obj = copy.deepcopy(params)
params_custom_obj["objective"] = loglikelihood

gbm = lgb.train(
    params_custom_obj, lgb_train, num_boost_round=10, init_model=gbm, feval=binary_error, valid_sets=lgb_eval
)

print("Finished 40 - 50 rounds with self-defined objective function and eval metric...")


# another self-defined eval metric
# f(preds: array, train_data: Dataset) -> metric_name: str, metric_value: float, maximize: bool
# accuracy
# NOTE: when you do customized loss function, the default prediction value is margin
# This may make built-in evaluation metric calculate wrong results
# For example, we are doing log likelihood loss, the prediction is score before logistic transformation
# Keep this in mind when you use the customization
def accuracy(preds, train_data):
    labels = train_data.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    return "accuracy", np.mean(labels == (preds > 0.5)), True


# Pass custom objective function through params
params_custom_obj = copy.deepcopy(params)
params_custom_obj["objective"] = loglikelihood

gbm = lgb.train(
    params_custom_obj,
    lgb_train,
    num_boost_round=10,
    init_model=gbm,
    feval=[binary_error, accuracy],
    valid_sets=lgb_eval,
)

print("Finished 50 - 60 rounds with self-defined objective function and multiple self-defined eval metrics...")

print("Starting a new training job...")


# callback
def reset_metrics():
    def callback(env):
        lgb_eval_new = lgb.Dataset(X_test, y_test, reference=lgb_train)
        if env.iteration - env.begin_iteration == 5:
            print("Add a new valid dataset at iteration 5...")
            env.model.add_valid(lgb_eval_new, "new_valid")

    callback.before_iteration = True
    callback.order = 0
    return callback


gbm = lgb.train(params, lgb_train, num_boost_round=10, valid_sets=lgb_train, callbacks=[reset_metrics()])

print("Finished first 10 rounds with callback function...")
