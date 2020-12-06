import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import filtfilt
from scipy.io import wavfile
from scipy.fft import fftshift
from pydub import AudioSegment


def plot_spectrogram(sound, sampling, name):
    f, t, spectre = signal.spectrogram(x=sound, fs=sampling, return_onesided=False)
    plt.pcolormesh(t, fftshift(f), np.log(fftshift(spectre)), shading='gouraud')
    plt.title(name)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def plot(x, y=None, name=None):
    if name is not None:
        plt.title(name)
    if y is not None:
        plt.plot(x, y)
    else:
        plt.plot(x)
    plt.show()


def convert_to_mp3(inp):
    src = inp + ".wav"
    dst = inp + ".mp3"

    sound = AudioSegment.from_wav(src)
    sound.export(dst, format="mp3")


def convert_to_wav(inp):
    src = inp + ".mp3"
    dst = inp + ".wav"

    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")


def create_watermark():
    matr = np.zeros((window_size, window_size), dtype=np.int32)

    for i in range(num_of_steps):
        temp_a = np.atleast_2d(voice[i * step:i * step + window_size])
        temp = temp_a.T @ temp_a
        matr += temp

    return np.linalg.eig(matr)[1][np.argmin(np.linalg.eig(matr)[0])]


def mark_signal():
    plot_spectrogram(voice, rate, 'Start signal spectrogram')
    plot(voice, name='Start signal')

    [b, a] = signal.iirfilter(10, 0.5, btype="highpass")

    z, p, k = signal.tf2zpk(b, a)
    b, a = signal.zpk2tf(z * 0.1, p * 0.01, k * 500)

    process_signal = signal.filtfilt(b, a, voice)

    w, h = signal.freqz(b, a)
    plot(w / np.pi, np.abs(h), 'IIR filter')

    w, h = signal.freqz(a, b)
    plot(w / np.pi, np.abs(h), 'Reverse IIR filter')

    plot_spectrogram(process_signal, rate, 'Filtered signal spectrogram')
    plot(process_signal, name='Filtered signal')

    process_signal[100000:100000 + window_size] += watermark * 30000
    print("Added watermark on pos 100000\n")

    process_signal = signal.filtfilt(a, b, process_signal).astype(np.int16)

    plot_spectrogram(process_signal, rate, 'Restored signal spectrogram')
    plot(process_signal, name='Restored signal')

    return process_signal, b, a


def check_correlation(voice):
    corr = np.correlate(voice, watermark, "valid")
    filt_corr = np.correlate(signal.filtfilt(b, a, voice).astype(np.int16), watermark, "valid")

    print("Dignal correlation peak:" + str(corr.argmax()))
    print("Filtered signal correlation peak:" + str(filt_corr.argmax()))

    return corr, filt_corr


rate, voice = wavfile.read("../HW7/sound.wav")

# используется сигнал с 2 каналами, поэтому вытаскиваю только 1
voice = np.moveaxis(voice, 1, 0)[0][100000:500000]

window_size = np.int(rate * 0.01)
step = np.int(window_size * 0.7)
num_of_steps = (voice.shape[0] - window_size) // step

watermark = create_watermark()

marked_signal, b, a = mark_signal()

corr, filt_corr = check_correlation(marked_signal)

plot(corr, name='Correlation')
plot(filt_corr, name='Correlation after filter')


noise = np.random.normal(0, 5, len(marked_signal)).astype(np.int16)
marked_signal += noise
print("\nAdded noise.")
check_correlation(marked_signal)

wavfile.write("marked.wav", rate, marked_signal)
convert_to_mp3("marked")
convert_to_wav("marked")

print("\nConverted: wav->mp3->wav.")
_, marked_signal = wavfile.read("marked.wav")
check_correlation(marked_signal)
