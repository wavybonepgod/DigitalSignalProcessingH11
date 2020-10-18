from scipy import fft as sf
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, title):
    plt.figure()
    plt.title(title)
    plt.plot(x, y)
    plt.show()


def fourier(sound, name):
    sound_amplitude = np.abs(sf.fft(window * sound))

    freq = sf.fftfreq(window_size, d=1 / rate)

    plot(np.arange(window_size) / rate, sound, "Original signal of " + name.lower())
    plot(freq, sound_amplitude / window_size, "Spectral amplitude of " + name.lower())

    if name == "Noise":
        print(name + " spectrum peak frequency: " + str(int(freq[sound_amplitude.argmax(axis=0)])))


def spectrogram(sound):
    f, t, spectre = signal.spectrogram(x=sound, fs=rate, window=window, noverlap=window_size - (window_size // 10))

    plt.pcolormesh(t, f, np.log(spectre), shading='gouraud')
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


wav = wavfile.read("voice.wav")

rate = wav[0]
voice = wav[1]
window_size = 20000

window = signal.windows.hamming(window_size)

fourier(voice[30000:30000 + window_size], "Noise")
fourier(voice[80000:80000 + window_size], "Silence")

spectrogram(voice[:150000])
