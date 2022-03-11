import numpy as np
import librosa
import scipy.io
import matplotlib.pyplot as plt

_, wave1 = scipy.io.wavfile.read("./data/Female_8k.wav")
_, wave2 = scipy.io.wavfile.read("./data/Male_8k.wav")
figure, ax = plt.subplots(nrows=2)
ax[0].plot(wave1[3500:4800])
ax[1].plot(wave2[3500:4500])
plt.show()




