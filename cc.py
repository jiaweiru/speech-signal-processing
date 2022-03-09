import numpy as np
import librosa
import scipy
import matplotlib.pyplot as plt

rec = scipy.signal.windows.boxcar(60)
rec_f = scipy.fft.fft(rec, 1024)
rec_f = rec_f / rec_f[0]

ham = scipy.signal.windows.hamming(60)
ham_f = scipy.fft.fft(ham, 1024)
ham_f = ham_f / ham_f[0]

figure, axes = plt.subplots(nrows=4)
x = np.arange(1024)
x = x / 1024

axes[0].plot(rec)
axes[0].set_title("rectangular", y=1)
axes[1].plot(x[:513], 20 * np.log10(np.abs(rec_f))[:513])
axes[1].set_ylabel("(dB)")
axes[2].plot(ham)
axes[2].set_title("hamming", y=1)
axes[3].plot(x[:513], 20 * np.log10(np.abs(ham_f))[:513])
axes[3].set_ylabel("(dB)")
figure.tight_layout()
figure.show()


