import numpy as np
from bitarray import bitarray
from scipy.io import wavfile
from pydub import AudioSegment


def get_u():
    A = 0.4

    t = np.linspace(0, 1, step)
    spl = np.piecewise(t, [(0 <= t) & (t <= 1 / 3), (1 / 3 <= t) & (t < 2 / 3), (2 / 3 <= t) & (t <= 1)],
                       [lambda x: 4.5 * x ** 2, lambda x: -9 * x ** 2 + 9 * x - 1.5,
                        lambda x: 4.5 - 9 * x + 4.5 * x ** 2])

    return 1 - A * spl, 1 + A * spl


def set_watermark():
    marked = signal.copy()

    u_zero, u_one = get_u()

    for i, bit in enumerate(watermark_bits):
        u = u_zero if bit == '0' else u_one
        marked[i * step:(i + 1) * step] = np.around(marked[i * step:(i + 1) * step] * u).astype(np.int16)
    return marked


def get_watermark():
    watermark = []

    for i in range(signal.shape[0] // step):
        diff = np.linalg.norm(marked_signal[i * step:(i + 1) * step]) - np.linalg.norm(signal[i * step:(i + 1) * step])
        diff = 1 if diff > 0 else 0
        watermark.append(diff)

    return watermark


def bits2a(b):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(b)]*8))


def convert_to_mp3():
    src = "marked.wav"
    dst = "marked.mp3"

    sound = AudioSegment.from_wav(src)
    sound.export(dst, format="mp3")


def convert_to_wav():
    src = "marked.mp3"
    dst = "marked.wav"

    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")


watermark_bits = bitarray()
watermark_bits.frombytes(b'   Never gonna give you up Never gonna let you down Never gonna run around and desert you')
watermark_bits = watermark_bits.to01()

rate, signal = wavfile.read("../HW7/sound.wav")

# используется сигнал с 2 каналами, поэтому вытаскиваю только 1
signal = np.moveaxis(signal, 1, 0)[0]   

step = 1000

marked_signal = set_watermark()
wavfile.write("marked.wav", rate, marked_signal)

restored_watermark = ''.join(str(i) for i in get_watermark())

print("Source watermark bits: ", watermark_bits)
print("Restored watermark bits: ", restored_watermark[:len(watermark_bits)])
print("Restored watermark text: ", bits2a(restored_watermark[:len(watermark_bits)]))

noise = np.random.normal(0, 5, len(marked_signal)).astype(np.int16)

marked_signal += noise

restored_watermark = ''.join(str(i) for i in get_watermark())

print("Restored watermark bits with noise: ", restored_watermark[:len(watermark_bits)])
print("Restored watermark text with noise: ", bits2a(restored_watermark[:len(watermark_bits)]))

convert_to_mp3()
convert_to_wav()

_, marked_signal = wavfile.read("marked.wav")

restored_watermark = ''.join(str(i) for i in get_watermark())

print("Restored watermark bits after convert: ", restored_watermark[:len(watermark_bits)])
print("Restored watermark text after convert: ", bits2a(restored_watermark[:len(watermark_bits)]))

print("Final: в начале моего файла есть несколько нулевых байтов, поэтому добавил пробелы,\n",
      "водяной знак успешно находит после конвертаций и после добавления шума, но с некоторыми дефектами")
