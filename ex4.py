import scipy.signal
import scipy.fft
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


class SpeechSignal:
    def __init__(self, path):
        self.sr, self.wave = scipy.io.wavfile.read(path)
        print("Data loading completed")

    def get_frame(self, win_l, start):
        frame = self.wave[start:start + win_l]
        win = scipy.signal.windows.hamming(win_l)
        frame_win = frame * win
        return frame_win

    def lpc_coeff(self, s, p, n_fft):
        s_pad = np.zeros(n_fft)
        s_pad[:s.shape[0]] = s[:]
        n = len(s_pad)
        # r(i), i = 1 ~ p - 1
        Rp = np.zeros(p + 1)
        for i in range(p + 1):
            Rp[i] = np.sum(np.multiply(s_pad[i:n], s_pad[:n - i]))

        # index[0] -> i = 1, index[p - 1] or [p - 1, p - 1]-> i = p
        Ep = np.zeros((p, 1))
        k = np.zeros((p, 1))
        a = np.zeros((p, p))

        # i=0
        Ep0 = Rp[0]

        # i=1
        k[0] = Rp[1] / Rp[0]
        a[0, 0] = k[0]
        Ep[0] = (1 - k[0] * k[0]) * Ep0

        if p > 1:
            # index = 1 ~ p - 1 -> 2 ~ p
            for m in range(1, p):
                k[m] = (Rp[m + 1] - np.sum(np.multiply(a[:m, m - 1], Rp[m:0:-1]))) / Ep[m - 1]
                a[m, m] = k[m]
                for j in range(m - 1, -1, -1):
                    a[j, m] = a[j, m - 1] - k[m] * a[m - j - 1, m - 1]
                Ep[m] = (1 - k[m] * k[m]) * Ep[m - 1]
        ar = np.zeros(p + 1)
        ar[0] = 1
        ar[1:] = -a[:, p - 1]
        G = np.sqrt(Ep[p - 1])
        return ar, G

    def plot_spec(self, s, ar, G, n_fft):
        origin_spec = np.abs(scipy.fft.fft(s, n_fft))
        origin_logspec = 20 * np.log10(origin_spec)

        lpa_spec = G / np.abs(scipy.fft.fft(ar, n_fft))
        lpa_logspec = 20 * np.log10(lpa_spec)

        fig, ax = plt.subplots()
        x = np.arange(n_fft)

        # sr = 8000
        x = x / n_fft * 8000
        ax.plot(x, origin_logspec, label='orgin', linewidth=0.5)
        ax.plot(x, lpa_logspec, label='lpa', linewidth=0.5)
        ax.set_xlim([0, 4000])
        ax.set_xlabel("Hz")
        ax.set_ylabel("dB")
        ax.legend()
        plt.show()

    def plot_waveform(self, s):
        fig, ax = plt.subplots()
        ax.plot(s)
        plt.show()


if __name__ == '__main__':
    wav_path = "./data/Male_8k.wav"
    nfft = 512
    p = 400

    speech = SpeechSignal(wav_path)
    f = speech.get_frame(512, 6300)
    ar, G = speech.lpc_coeff(f, p, nfft)
    speech.plot_waveform(f)
    speech.plot_spec(f, ar, G, nfft)


