from stats import extract_from_folder
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    folder = 'data'
    subfolder = 'quasi-static'
    extension = '.npz'
    max_files = 100
    chunk_size = 1000
    n_pca = 10

    features_quasi = extract_from_folder(folder, subfolder, extension, max_files, shuffle=False, chunk_size=chunk_size,
                                         n_pca=n_pca).to_numpy()

    labels_quasi = np.zeros(features_quasi.shape[1])

    subfolder = 'moving'

    features_moving = extract_from_folder(folder, subfolder, extension, max_files, shuffle=False, chunk_size=chunk_size,
                                          n_pca=n_pca).to_numpy()

    labels_moving = np.ones(features_moving.shape[1])

    X = np.hstack((features_quasi, features_moving)).T
    y = np.hstack((labels_quasi, labels_moving))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)

    # TODO Grid search
    # https://xgboost.readthedocs.io/en/stable/parameter.html
    # We need to define parameters as dict
    params = {
        "learning_rate": 0.3,
        "max_depth": 3
    }
    # training, we set the early stopping rounds parameter
    bst = xgb.train(params, dtrain, evals=[(dtrain, "train"), (dtest, "validation")],
                    num_boost_round=1000, early_stopping_rounds=20)
    # make prediction
    preds_test = np.round(bst.predict(dtest))

    acc = accuracy_score(y_test, preds_test)
    print('Accuracy ', acc)

    # bst.save_model('model_file_name_{:f}_acc.json'.format(acc))
