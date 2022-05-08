import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def my_fft(x, Fs, donwsample_factor=1, Nfft=1024, window_type=None):
    x_downsampled = x[::donwsample_factor]

    if window_type is None:
        window = np.ones(len(x_downsampled))
    elif window_type == 'hamming':
        window = np.hamming(len(x_downsampled))
    else:
        raise NotImplementedError()

    fourier = np.fft.fft(x_downsampled * window, n=Nfft)
    amp = np.abs(fourier * 2 / len(x_downsampled))
    phase = np.angle(fourier)

    faxis = np.fft.fftfreq(n=Nfft) * Fs / donwsample_factor

    return amp, phase, faxis


if __name__ == '__main__':
    """
    0. Do we bandpass filter before hand? Yeah, why not?
    1. Downsample signal to get Fs to the 0-5 Hz (or less) range for better sampling of the spectrum
    2. Window that signal to kill the side lobes
    3. Calculate fft and abs(fft)
    4. Find peaks
    5. Sort peaks by height
    6. Take the highest peaks (2 or how many you expect)
    7. Read frequencies
    """
    Fs = 1000
    t = np.arange(start=0, stop=3, step=1 / Fs)
    f = [0.5, 2]
    a = [1, 4]
    x = 0
    # TODO add noise. Or just look at the real signal
    for i in range(len(f)):
        x += a[i] * np.sin(2 * np.pi * f[i] * t)

    flim = 3

    plt.figure()
    plt.plot(t, x)
    plt.show()

    # TODO what does range fft do? I don't think it does fft just in a small range
    donwsample_factor = 100
    Nfft = 2 ** 14

    amp, phase, faxis = my_fft(x, Fs, donwsample_factor, Nfft, window_type='hamming')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(faxis, amp)
    ax[0].set_title('Amp')
    ax[0].set_xlim(left=-flim, right=flim)
    ax[1].plot(faxis, phase)
    ax[1].set_title('Phase')
    ax[1].set_xlim(left=-flim, right=flim)

    fig.tight_layout()
    fig.show()

    right_lim = int(np.floor(Nfft / 2))
    right_side_amp = amp[:right_lim]
    peaks = signal.find_peaks(right_side_amp)

    sorted_peak_indices = np.argsort(right_side_amp[peaks[0]])[::-1]
    sorted_peaks = peaks[0][sorted_peak_indices]

    highest_peaks = sorted_peaks[:2]

    plt.figure()
    plt.plot(faxis[:right_lim], right_side_amp)
    plt.scatter(faxis[highest_peaks], right_side_amp[highest_peaks], color='r')
    plt.show()

    print('Peak frequencies: ', faxis[highest_peaks], ' Hz')
