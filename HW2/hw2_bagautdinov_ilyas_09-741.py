import numpy as np
import matplotlib.pyplot as plt


def get_signal(signal_time, fs, freq):
    signal_periods = np.linspace(0, signal_time, signal_time * fs)
    harmonic = np.sin(2 * np.pi * signal_periods * freq)
    return signal_periods, harmonic


def recover_signal(signal_periods, harmonic, fs, analog_periods):
    delta = np.tile(analog_periods, [len(signal_periods), 1]).T - np.tile(signal_periods, [len(analog_periods), 1])
    recovered = np.sum(np.tile(harmonic, [len(analog_periods), 1]) * np.sinc(delta * fs), axis=1)
    return recovered


def plot_signal(analog_periods, harmonic_orig, harmonic_rec):
    plt.plot(analog_periods, harmonic_orig)
    plt.plot(analog_periods, harmonic_rec)

    plt.show()


time = 1
frequency = 30

fs_orig = 20000
fs_nice = 100
fs_bad = 15

orig_signal_periods, orig_harmonic = get_signal(time, fs_orig, frequency)
nice_signal_periods, nice_harmonic = get_signal(time, fs_nice, frequency)
bad_signal_periods, bad_harmonic = get_signal(time, fs_bad, frequency)

nice_recovered = recover_signal(nice_signal_periods, nice_harmonic, fs_nice, orig_signal_periods)
bad_recovered = recover_signal(bad_signal_periods, bad_harmonic, fs_bad, orig_signal_periods)

plot_signal(orig_signal_periods, orig_harmonic, nice_recovered)
plot_signal(orig_signal_periods, orig_harmonic, bad_recovered)
