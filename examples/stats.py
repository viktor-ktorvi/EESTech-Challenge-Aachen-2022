from utils import get_filenames, load_radar_dict
import os
import pandas as pd


if __name__ == '__main__':
    folder = 'data'
    subfolder = 'quasi-static'
    extension = '.npz'
    max_files = 3
    filenames = get_filenames(folder=folder, subfolder=subfolder, extension='.npz', max_files=max_files, shuffle=False)

    radar_dicts = [load_radar_dict(os.path.join(folder, subfolder), filename) for filename in filenames]

    a = dict2pandas_compatible_dict(radar_dicts[0])
