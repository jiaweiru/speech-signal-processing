import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy
import numpy as np


class SpeechSignal:
    def __init__(self, path):
        self.wave, self.sr = librosa.load(path, sr=None)
        print("Data loading completed")

    def plotter_p(self, win_type="rec", hop_length=6, plot_num=5, frame_len_start=64):
        fig, ax = plt.subplots(nrows=plot_num + 1)
        fig.suptitle("{x}, hop length={y}".format(x=win_type, y=hop_length))
        for i in range(plot_num):
            frame_length = frame_len_start * (2 ** i)
            dic = {"rec": scipy.signal.windows.boxcar(frame_length), "ham": scipy.signal.windows.hamming(frame_length)}
            win = dic[win_type]
            win = win.reshape(frame_length, 1)

            frame = librosa.util.frame(self.wave, frame_length=frame_length, hop_length=hop_length)
            frame_win = win * frame
            power = np.mean(frame_win * frame_win, axis=0)

            ax[i].plot(power)
            # librosa.display.waveshow(frame, sr=self.sr, max_points=self.sr // 2, ax=ax[i])
            ax[i].set_title("{x},M = {m}".format(x=win_type, m=frame_length), y=0.5)
            ax[i].xaxis.set_visible(False)
        ax[plot_num].plot(self.wave)
        # librosa.display.waveshow(self.wave, sr=self.sr, max_points=self.sr // 2, ax=ax[plot_num])
        fig.show()

    def plotter_f(self, win_type="rec", hop_length=6, plot_num=5, frame_len_start=64, time_index=None):
        fig, ax = plt.subplots(nrows=plot_num + 1)
        for i in range(plot_num):
            frame_length = frame_len_start * (2 ** i)
            dic = {"rec": scipy.signal.windows.boxcar(frame_length),
                   "ham": scipy.signal.windows.hamming(frame_length)}
            win = dic[win_type]
            spec = librosa.stft(self.wave, n_fft=frame_length, hop_length=hop_length, win_length=frame_length,
                                window=win)

            if time_index:
                frame_index = int(time_index * self.sr/hop_length)
                spec_frame = np.abs(spec)[:, frame_index]
                ax[i].plot(spec_frame)
            else:
                spec_maglog = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
                librosa.display.specshow(spec_maglog, sr=self.sr, hop_length=hop_length, ax=ax[i])
        librosa.display.waveshow(self.wave, sr=self.sr, max_points=self.sr // 2, ax=ax[plot_num])
        fig.show()


if __name__ == '__main__':
    wav_path = "./data/Male_8k.wav"
    speech = SpeechSignal(wav_path)
    # speech.plotter_p(win_type="ham", hop_length=hop_len, plot_num=5, frame_len_start=64)
    speech.plotter_f(win_type="ham", hop_length=128, plot_num=3, frame_len_start=64, time_index=2.5)
