import numpy as np


def get_quanted(quant_levels, signal):
    quanted_index = np.searchsorted(quant_levels, signal)
    quanted_index[quanted_index == np.size(quant_levels)-1] = np.size(quant_levels)-2
    mask = np.abs(quant_levels[quanted_index] - signal) < np.abs(quant_levels[quanted_index + 1] - signal)
    quanted = np.where(mask, quant_levels[quanted_index], quant_levels[quanted_index + 1])
    return quanted


quant = 2**16
fs = 1000
count = 10
frequency = 10

signal = np.linspace(0, count, count*fs)

sin_quant_levels = np.linspace(-1, 1, quant)
random_quant_levels = np.linspace(0, 1, quant)

sin_signal = np.sin(2 * np.pi * signal * frequency)
random_signal = np.random.rand(count * fs)

quanted_sin_signal = get_quanted(sin_quant_levels, sin_signal)
quanted_random_signal = get_quanted(random_quant_levels, random_signal)

sin_signal_error = sin_signal - quanted_sin_signal
random_signal_error = random_signal - quanted_random_signal

sin_snr = 10 * np.log10(np.var(sin_signal)/np.var(sin_signal_error))
random_snr = 10 * np.log10(np.var(random_signal)/np.var(random_signal_error))

print("Sine snr: ", sin_snr,
      "\nRandom snr: ", random_snr,
      "\nTheoretical snr: 88.8 ")
