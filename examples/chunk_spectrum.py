from stats import get_chunks
from utils import load_as_dataframe, get_filenames
from spectrum import my_fft
from matplotlib import pyplot as plt
import os
import numpy as np

if __name__ == '__main__':
    folder = 'data'
    subfolder = 'activity-anxionsness'
    extension = '.npz'
    max_files = 1

    filenames = get_filenames(folder=folder, subfolder=subfolder, extension=extension, max_files=max_files,
                              shuffle=True)
    dfs = [load_as_dataframe(os.path.join(folder, subfolder), f) for f in filenames]

    chunk_size = 3000
    Fs = 1000
    donwsample_factor = 100
    Nfft = 2 ** 8
    right_lim = int(np.floor(Nfft / 2))

    chunks = get_chunks(dfs, chunk_size, subtract_mean=True)

    chunk = chunks[20]

    # TODO Filter signal

    # time
    fig_time, ax_time = plt.subplots(len(chunk.columns), 1)

    fig_time.suptitle('Time')

    for i, column in enumerate(chunk.columns):
        ax_time[i].set_title(column)
        ax_time[i].plot(chunk[column].to_numpy())

    fig_time.tight_layout()

    fig_time.show()

    # Frequency
    fig_fft, ax_fft = plt.subplots(len(chunk.columns), 1)

    fig_fft.suptitle('Frequency')

    for i, column in enumerate(chunk.columns):

        amp, phase, faxis = my_fft(chunk[column].to_numpy(), Fs, donwsample_factor, Nfft, window_type='hamming')
        right_side_amp = amp[:right_lim]
        right_faxis = faxis[:right_lim]

        ax_fft[i].set_title(column)
        ax_fft[i].plot(right_faxis, right_side_amp)

    fig_fft.tight_layout()

    fig_fft.show()

