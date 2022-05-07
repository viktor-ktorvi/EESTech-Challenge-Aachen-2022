from stats import extract_from_folder
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

import datetime


def extract_from_subfolders(folder, subfolders, extension, max_files, chunk_size, n_pca, label_val):
    features = None
    labels = None
    for i in range(len(subfolders)):
        if i == 0:
            features = extract_from_folder(folder, subfolders[i], extension, max_files, shuffle=False,
                                           chunk_size=chunk_size,
                                           n_pca=n_pca).to_numpy()

            labels = np.zeros(features.shape[1]) + label_val
        else:
            features = np.concatenate((
                extract_from_folder(folder, subfolders[i], extension, max_files, shuffle=False, chunk_size=chunk_size,
                                    n_pca=n_pca).to_numpy(),
                features
            ), axis=1)
            labels = np.concatenate((
                np.zeros(features.shape[1] - len(labels)) + label_val,
                labels
            ))

    return features, labels


if __name__ == '__main__':
    folder = 'data'
    subfolders = ['quasi-static', 'activity-anxionsness', 'relaxed-after-activity']
    extension = '.npz'

    hyperparams = {
        'chunk_size': 1000,
        'n_pca': 10,
        'num_boost_round': 1000,
        'early_stopping_rounds': 20
    }

    max_files = 100
    chunk_size = hyperparams['chunk_size']
    n_pca = hyperparams['n_pca']

    load_old_data = True
    data_savepath = 'train_data.npz'

    if load_old_data:
        data_dict = np.load(data_savepath)
        X = data_dict['X']
        y = data_dict['y']
    else:

        features_quasi, labels_quasi = extract_from_subfolders(folder, subfolders, extension, max_files, chunk_size,
                                                               n_pca,
                                                               label_val=0)

        subfolders = ['moving']

        features_moving, labels_moving = extract_from_subfolders(folder, subfolders, extension, max_files, chunk_size,
                                                                 n_pca,
                                                                 label_val=1)

        X = np.concatenate((features_quasi, features_moving), axis=1).T
        y = np.concatenate((labels_quasi, labels_moving))

        np.savez(data_savepath, X=X, y=y)

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
                    num_boost_round=hyperparams['num_boost_round'],
                    early_stopping_rounds=hyperparams['early_stopping_rounds'])
    # make prediction
    # ntree_limit should use the optimal number of trees https://mljar.com/blog/xgboost-save-load-python/
    preds_test = np.round(bst.predict(dtest, ntree_limit=bst.best_ntree_limit))

    acc = accuracy_score(y_test, preds_test)
    print('Accuracy ', acc)

    datetime_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    savepath = 'boost_acc_{:d}_'.format(round(acc * 100)) + datetime_now
    os.mkdir(savepath, 0o666)

    with open(os.path.join(savepath, 'hyperparams.json'), 'w') as fp:
        json.dump(hyperparams, fp)

    params.update({'best_ntree_limit': bst.best_ntree_limit})

    with open(os.path.join(savepath, 'train_params.json'), 'w') as fp:
        json.dump(params, fp)

    bst.save_model(os.path.join(savepath, "model.json"))
