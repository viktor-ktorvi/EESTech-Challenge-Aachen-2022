import numpy as np

from utils import get_filenames, load_as_dataframe
from stats import get_chunks
import os
import pandas as pd
from spectrum import butter_bandpass
from chunk_spectrum import filter_dataframe, chunk_fft
from tqdm import tqdm


def extract_stress_features(dfs, chunk_size, Fs, donwsample_factor, Nfft, window_type):
    chunks = get_chunks(dfs, chunk_size, subtract_mean=True)

    b_heart, a_heart = butter_bandpass(lowcut=0.8, highcut=2, fs=Fs, order=3)
    b_breath, a_breath = butter_bandpass(lowcut=0.2, highcut=0.5, fs=Fs, order=3)

    feature_list = []
    for chunk in tqdm(chunks):
        heart_chunk = filter_dataframe(b_heart, a_heart, chunk)
        heart_chunk_downsampled = heart_chunk[::donwsample_factor]

        breath_chunk = filter_dataframe(b_breath, a_breath, chunk)
        breath_chunk_downsampled = breath_chunk[::donwsample_factor]

        h_freqs_dict = chunk_fft(heart_chunk_downsampled, Fs, donwsample_factor, Nfft, window_type,
                                 columns=heart_chunk_downsampled.columns,
                                 show=False,
                                 num_peaks=1, title='Heart')

        b_freqs_dict = chunk_fft(breath_chunk_downsampled, Fs, donwsample_factor, Nfft, window_type,
                                 columns=breath_chunk_downsampled.columns,
                                 show=False,
                                 num_peaks=1, title='Breath')

        feature_list.append(np.concatenate(
            (pd.DataFrame.from_dict(h_freqs_dict).to_numpy(), pd.DataFrame.from_dict(b_freqs_dict).to_numpy()), axis=1))

    feature_matrix = np.concatenate(feature_list)

    return feature_matrix


def extract_stress_from_folder(folder, subfolder, extension, max_files, shuffle, chunk_size, Fs, donwsample_factor,
                               Nfft, window_type):
    filenames = get_filenames(folder=folder, subfolder=subfolder, extension=extension, max_files=max_files,
                              shuffle=shuffle)
    dfs = [load_as_dataframe(os.path.join(folder, subfolder), f) for f in filenames]

    return extract_stress_features(dfs, chunk_size, Fs, donwsample_factor, Nfft, window_type)


def extract_stress_from_subfolders(folder, subfolders, extension, max_files, shuffle, chunk_size, Fs, donwsample_factor,
                                   Nfft, window_type):
    matrices = []
    for subfolder in subfolders:
        matrices.append(extract_stress_from_folder(folder, subfolder, extension, max_files, shuffle, chunk_size, Fs,
                                                   donwsample_factor,
                                                   Nfft, window_type))

    return np.concatenate(matrices)


if __name__ == '__main__':
    folder = 'data'
    subfolder = 'activity-anxionsness'
    extension = '.npz'
    max_files = 100
    shuffle = False
    chunk_size = 3000
    Fs = 1000
    donwsample_factor = 100
    Nfft = 2 ** 9
    window_type = 'hamming'
    relaxed_subfolders = ['relaxed-after-activity']

    relaxed_features = extract_stress_from_subfolders(folder, relaxed_subfolders, extension, max_files, shuffle,
                                                      chunk_size, Fs,
                                                      donwsample_factor,
                                                      Nfft, window_type)

    active_subfolders = ['activity-anxionsness']
    active_features = extract_stress_from_subfolders(folder, active_subfolders, extension, max_files, shuffle,
                                                     chunk_size, Fs,
                                                     donwsample_factor,
                                                     Nfft, window_type)
