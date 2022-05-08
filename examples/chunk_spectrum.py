from stats import get_chunks
from utils import load_as_dataframe, get_filenames

import os

if __name__ == '__main__':
    folder = 'data'
    subfolder = 'activity-anxionsness'
    extension = '.npz'
    max_files = 1

    filenames = get_filenames(folder=folder, subfolder=subfolder, extension=extension, max_files=max_files,
                              shuffle=True)
    dfs = [load_as_dataframe(os.path.join(folder, subfolder), f) for f in filenames]

    chunk_size = 3000

    chunks = get_chunks(dfs, chunk_size, subtract_mean=True)
