import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def plot_spectrogram(sound, sampling, name):
    f, t, spectre = signal.spectrogram(x=sound, fs=sampling)

    plt.pcolormesh(t, f, spectre, shading='gouraud')
    plt.title(name)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def plot_signal(x1, y1, x2, y2):
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.show()


def plot(x, y, name):
    plt.title(name)
    plt.plot(x, y)
    plt.show()


def filter_signal(process_signal):
    [b, a] = [signal.firwin(25, [0.4, 0.65]), 1]
    process_signal = signal.filtfilt(b, a, process_signal)
    w, h = signal.freqz(b, a)
    plot(w / np.pi, np.abs(h), 'FIR filter')

    h_temp = h

    [b, a] = signal.iirfilter(10, [0.25, 0.8], btype="bandpass")
    process_signal = signal.filtfilt(b, a, process_signal)
    w, h = signal.freqz(b, a)
    plot(w / np.pi, np.abs(h), 'IIR filter')

    plot(w / np.pi, np.abs(h * h_temp), 'Result filter')

    return process_signal


fs = 200
count = 3
frequency_arr = [15, 30, 50, 75, 90]
size = count * fs

sum_signal = np.zeros(size)
lin_signal = np.linspace(0, count, size)

for frequency in frequency_arr:
    sum_signal += np.sin(2 * np.pi * lin_signal * frequency)

filtered = filter_signal(sum_signal)

plot_signal(lin_signal[:200], sum_signal[:200], lin_signal[:200], filtered[:200])
plot_spectrogram(sum_signal, fs, 'Start signal spectrogram')
plot_spectrogram(filtered, fs, 'Filtered signal spectrogram')
