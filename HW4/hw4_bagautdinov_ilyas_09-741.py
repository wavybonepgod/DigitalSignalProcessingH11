import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as pol
from scipy.io import wavfile
from scipy.signal import correlate

voice = wavfile.read("watermark.wav")[1]

s = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape((-1, 1))
d = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

companion_matrix = pol.polycompanion([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]).T.astype(int)

water = np.zeros(2 ** 11 - 2, dtype=int)

for i in range(1, 2 ** 11 - 1):
    s = companion_matrix @ s
    w = d @ s % 2
    if w == 0:
        w = -1
    water[i - 1] = w

corr = correlate(voice, water, mode='valid')

plt.figure()
plt.plot(corr)
plt.show()

print(corr.argmax())
