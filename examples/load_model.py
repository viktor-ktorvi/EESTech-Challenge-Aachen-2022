import xgboost as xgb
import os

from utils import load_as_dataframe, get_filenames

if __name__ == '__main__':
    model_folder = 'boost_acc_91_2022_05_07_17_08_30'
    model_xgb_2 = xgb.Booster()
    model_xgb_2.load_model(os.path.join(model_folder, "model.json"))

    folder = 'data'
    extension = '.npz'
    subfolder = 'quasi-static'
    filename = get_filenames(folder, subfolder=subfolder, extension=extension, max_files=1, shuffle=True)
    df = load_as_dataframe(os.path.join(folder, subfolder), filename[0])
