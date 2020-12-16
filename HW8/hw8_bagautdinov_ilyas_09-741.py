import numpy as np
from scipy import fft as sf
from bitarray import bitarray
from scipy.io import wavfile
import matplotlib.pyplot as plt


def set_watermark():
    marked = signal.copy()

    u_zero = 1/(1-eps)
    u_one = 1/(1+eps)

    for i, bit in enumerate(watermark_bits):
        u = u_zero if bit == '0' else u_one

        fourier = sf.fft(marked[i * step:(i + 1) * step])

        orig_power = np.linalg.norm(fourier)

        fourier[1:3] /= u
        fourier[-2:] /= u

        new_power = np.linalg.norm(fourier)
        other_power = np.linalg.norm(fourier[2:-2])

        fourier[2:-2] *= np.sqrt((other_power + orig_power - new_power) / other_power)

        marked[i * step:(i + 1) * step] = sf.ifft(fourier).real

    return marked


def get_watermark():
    watermark = []

    for i in range(signal.shape[0] // step):
        diff = np.sum(np.abs(sf.fft(marked_signal[i * step:(i + 1) * step])[1:3]) -
                      np.abs(sf.fft(signal[i * step:(i + 1) * step])[1:3]))
        diff = 1 if diff > 0 else 0
        watermark.append(diff)

    return watermark


def bits2a(b):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(b)]*8))


eps = 0.005

watermark_bits = bitarray()
watermark_bits.frombytes(b'Never gonna give you up Never gonna let you down Never gonna run around and desert you')
watermark_bits = watermark_bits.to01()

rate, voice = wavfile.read("sound.wav")

# используется сигнал с 2 каналами, поэтому вытаскиваю только 1
signal = np.moveaxis(voice, 1, 0)[0][100000:500000]

step = np.int(np.ceil(np.log(rate * 0.01) / np.log(2)))

marked_signal = np.int16(set_watermark())

restored_watermark = ''.join(str(i) for i in get_watermark())

noise = np.random.normal(0, 5, len(marked_signal)).astype(np.int16)
marked_signal += noise
restored_watermark_noise = ''.join(str(i) for i in get_watermark())


print("Source watermark bits:   ", watermark_bits)
print("Restored watermark bits: ", restored_watermark[:len(watermark_bits)])
print("Restored watermark text: ", bits2a(restored_watermark[:len(watermark_bits)]))
print("Restored watermark bits with noise: ", restored_watermark_noise[:len(watermark_bits)])
print("Restored watermark text with noise: ", bits2a(restored_watermark_noise[:len(watermark_bits)]))

wavfile.write("marked.wav", rate, marked_signal)

plt.plot(signal)
plt.plot(marked_signal)
plt.show()
