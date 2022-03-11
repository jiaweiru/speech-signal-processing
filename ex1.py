import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy
import scipy.io
import numpy as np


class SpeechSignal:
    def __init__(self, path):
        self.sr, self.wave = scipy.io.wavfile.read(path)
        print("Data loading completed")

    def plotter_p(self, win_type="rec", plot_num=5, frame_len_start=64, factor=2):
        fig, ax = plt.subplots(nrows=plot_num + 1)
        for i in range(plot_num):
            frame_length = frame_len_start * (factor ** i)
            hop_length = frame_length // 2
            dic = {"rec": scipy.signal.windows.boxcar(frame_length), "ham": scipy.signal.windows.hamming(frame_length)}
            win = dic[win_type]

            # librosa.util.frame：no padding and cut off at the end
            frame = librosa.util.frame(self.wave, frame_length=frame_length, hop_length=hop_length, axis=0)
            frame_win = win * frame
            energy = np.mean(frame_win * frame_win, axis=1)
            energy_log = 10 * np.log10(energy)
            # energy_log = librosa.power_to_db(energy)

            ax[i].plot(energy)
            # librosa.display.waveshow(frame, sr=self.sr, max_points=self.sr // 2, ax=ax[i])
            ax[i].set_title(win_type + ", N = {m}".format(m=frame_length), y=1)
        ax[plot_num].plot(self.wave)
        ax[plot_num].set_title("speech signal", y=1)
        # librosa.display.waveshow(self.wave, sr=self.sr, max_points=self.sr // 2, ax=ax[plot_num])
        fig.tight_layout()
        plt.show()

    def plotter_f(self, win_type="rec", plot_num=5, frame_len_start=64, factor=2, time_index=None):
        fig, ax = plt.subplots(nrows=plot_num * 2)
        frame_len_end = frame_len_start * (factor ** (plot_num - 1))
        for i in range(0, plot_num * 2, 2):
            frame_length = frame_len_start * (factor ** (i // 2))
            hop_length = frame_length // 2
            dic = {"rec": scipy.signal.windows.boxcar(frame_length), "ham": scipy.signal.windows.hamming(frame_length)}
            win = dic[win_type]

            if time_index:
                start_index = int(time_index * self.sr)
                spec = scipy.fft.fft(self.wave[start_index:start_index + frame_length] * win)

                x = np.arange(frame_length) / frame_length * 8000

                # n_fft//2 + 1
                ax[i].plot(x[:frame_length // 2 + 1], 20 * np.log10(np.abs(spec[:frame_length // 2 + 1])))
                ax[i].set_title("{x},M = {m}".format(x=win_type, m=frame_length), y=1)
                ax[i].set_ylabel("(dB)")
                ax[i].set_xlabel("(Hz)")
                ax[i].set_xlim([0, self.sr // 2])

                wave_i = np.zeros(frame_len_end)
                wave_i[:frame_length] = self.wave[start_index:start_index + frame_length] * win
                ax[i+1].plot(wave_i)
            else:
                spec = librosa.stft(self.wave * 1.0, n_fft=frame_length, hop_length=hop_length, win_length=frame_length,
                                    window=win)
                spec_maglog = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
                librosa.display.specshow(spec_maglog, sr=self.sr, hop_length=hop_length, ax=ax[i])
        if time_index:
            pass
        else:
            ax[plot_num].plot(self.wave)
        # librosa.display.waveshow(self.wave * 1.0, sr=self.sr, max_points=self.sr // 2, ax=ax[plot_num])
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    wav_path = "./data/Speech_8k/S001.wav"
    speech = SpeechSignal(wav_path)
    # speech.plotter_p(win_type="rec", plot_num=4, frame_len_start=64)
    # speech.plotter_p(win_type="ham", plot_num=4, frame_len_start=64)

    # vioced
    # speech.plotter_f(win_type="ham", plot_num=1, factor=8, frame_len_start=256, time_index=2.5)
    time = [0.8, 0.85, 0.9, 0.95]
    for i in time:
        speech.plotter_f(win_type="ham", plot_num=1, factor=2, frame_len_start=512, time_index=i)

    # unvoiced
    # speech.plotter_f(win_type="rec", plot_num=1, factor=8, frame_len_start=64, time_index=2.08)
    # speech.plotter_f(win_type="ham", plot_num=2, factor=8, frame_len_start=64, time_index=2.08)
