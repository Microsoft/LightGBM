Python-package Examples
=======================

Here is an example for LightGBM to use Python-package.

You should install LightGBM [Python-package](https://github.com/lightgbm-org/LightGBM/tree/main/python-package) first.

You also need scikit-learn, pandas, matplotlib (only for plot example), and scipy (only for logistic regression example) to run the examples, but they are not required for the package itself. You can install them with pip:

```
pip install scikit-learn pandas matplotlib scipy -U
```

Now you can run examples in this folder, for example:

```
python simple_example.py
```

Examples include:

- [`dask/`](./dask): examples using Dask for distributed training
- [simple_example.py](https://github.com/lightgbm-org/LightGBM/blob/main/examples/python-guide/simple_example.py)
    - Construct Dataset
    - Basic train and predict
    - Eval during training
    - Early stopping
    - Save model to file
- [sklearn_example.py](https://github.com/lightgbm-org/LightGBM/blob/main/examples/python-guide/sklearn_example.py)
    - Create data for learning with sklearn interface
    - Basic train and predict with sklearn interface
    - Feature importances with sklearn interface
    - Self-defined eval metric with sklearn interface
    - Find best parameters for the model with sklearn's GridSearchCV
- [advanced_example.py](https://github.com/lightgbm-org/LightGBM/blob/main/examples/python-guide/advanced_example.py)
    - Construct Dataset
    - Set feature names
    - Directly use categorical features without one-hot encoding
    - Save model to file
    - Dump model to JSON format
    - Get feature names
    - Get feature importances
    - Load model to predict
    - Dump and load model with pickle
    - Load model file to continue training
    - Change learning rates during training
    - Change any parameters during training
    - Self-defined objective function
    - Self-defined eval metric
    - Callback function
- [logistic_regression.py](https://github.com/lightgbm-org/LightGBM/blob/main/examples/python-guide/logistic_regression.py)
    - Use objective `xentropy` or `binary`
    - Use `xentropy` with binary labels or probability labels
    - Use `binary` only with binary labels
    - Compare speed of `xentropy` versus `binary`
- [plot_example.py](https://github.com/lightgbm-org/LightGBM/blob/main/examples/python-guide/plot_example.py)
    - Construct Dataset
    - Train and record eval results for further plotting
    - Plot metrics recorded during training
    - Plot feature importances
    - Plot split value histogram
    - Plot one specified tree
    - Plot one specified tree with Graphviz
- [dataset_from_multi_hdf5.py](https://github.com/lightgbm-org/LightGBM/blob/main/examples/python-guide/dataset_from_multi_hdf5.py)
  - Construct Dataset from multiple HDF5 files
  - Avoid loading all data into memory
