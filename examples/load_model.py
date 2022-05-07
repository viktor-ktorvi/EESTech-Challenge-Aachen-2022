import xgboost as xgb
import os
import json

from utils import load_as_dataframe, get_filenames
from stats import extract_features

if __name__ == '__main__':
    model_folder = 'boost_acc_91_2022_05_07_17_08_30'
    bst = xgb.Booster()
    bst.load_model(os.path.join(model_folder, "model.json"))

    folder = 'data'
    extension = '.npz'
    subfolder = 'quasi-static'
    filename = get_filenames(folder, subfolder=subfolder, extension=extension, max_files=1, shuffle=True)
    df = load_as_dataframe(os.path.join(folder, subfolder), filename[0])

    with open(os.path.join(model_folder, 'hyperparams.json'), 'r') as f:
        hyperparams = json.load(f)

    with open(os.path.join(model_folder, 'train_params.json'), 'r') as f:
        train_params = json.load(f)

    feature = extract_features([df], chunk_size=hyperparams['chunk_size'], n_pca=hyperparams['n_pca'])


