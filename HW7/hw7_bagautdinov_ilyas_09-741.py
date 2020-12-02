import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fftshift


def plot_spectrogram(sound, sampling, name):
    f, t, spectre = signal.spectrogram(x=sound, fs=sampling, return_onesided=False)

    plt.pcolormesh(t, fftshift(f), np.log(fftshift(spectre)), shading='gouraud')
    plt.title(name)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def plot_signal(x1):
    plt.plot(x1)
    # plt.plot(x2, y2)
    plt.show()


rate, voice = wavfile.read("../HW7/sound.wav")

# используется сигнал с 2 каналами, поэтому вытаскиваю только 1
voice = np.moveaxis(voice, 1, 0)[0][100000:200000]

window_size = 5000
step = 4000
num_of_steps = (voice.shape[0] - window_size) // step

matr = np.zeros((window_size, window_size), dtype=np.int32)

for i in range(num_of_steps):
    temp_a = np.atleast_2d(voice[i * step:i * step + window_size])
    temp = temp_a.T @ temp_a
    matr += temp

watermark = matr[np.argmin(np.linalg.eig(matr)[0])]

plot_spectrogram(voice, rate, 'Start signal spectrogram')
plot_signal(voice)

[b, a] = signal.iirfilter(10, 0.5, btype="highpass")
process_signal = signal.filtfilt(b, a, voice)

plot_spectrogram(process_signal, rate, 'Start signal spectrogram')
plot_signal(process_signal)

a = 1