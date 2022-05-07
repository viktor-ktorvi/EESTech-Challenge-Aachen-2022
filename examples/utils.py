import os
import numpy as np


def radar2dict(radar_files):
    radar_files_dict = {}
    for r_file in radar_files.files:
        radar_files_dict[r_file] = radar_files[r_file]

    return radar_files_dict


def load_radar_dict(folder, filename):
    path = os.path.join(folder, filename)
    radar_files = np.load(path)

    return radar2dict(radar_files)
