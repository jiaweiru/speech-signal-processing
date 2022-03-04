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

    def plotter_p(self, win_type="rec", plot_num=5, frame_len_start=64):
        fig, ax = plt.subplots(nrows=plot_num + 1)
        fig.suptitle("{x}".format(x=win_type))
        for i in range(plot_num):
            frame_length = frame_len_start * (2 ** i)
            hop_length = frame_length // 2
            dic = {"rec": scipy.signal.windows.boxcar(frame_length), "ham": scipy.signal.windows.hamming(frame_length)}
            win = dic[win_type]
            win = win.reshape(frame_length, 1)

            frame = librosa.util.frame(self.wave, frame_length=frame_length, hop_length=hop_length)
            frame_win = win * frame
            power = np.sum(frame_win * frame_win, axis=0)
            power_log = np.log10(power)
            # power_log = librosa.power_to_db(power)

            ax[i].plot(power)
            # librosa.display.waveshow(frame, sr=self.sr, max_points=self.sr // 2, ax=ax[i])
            ax[i].set_title("{x},M = {m}".format(x=win_type, m=frame_length), y=0.5)
            ax[i].xaxis.set_visible(False)
        ax[plot_num].plot(self.wave)
        # librosa.display.waveshow(self.wave, sr=self.sr, max_points=self.sr // 2, ax=ax[plot_num])
        fig.show()

    def plotter_f(self, win_type="rec", plot_num=5, frame_len_start=64, time_index=None):
        fig, ax = plt.subplots(nrows=plot_num + 1)
        for i in range(plot_num):
            frame_length = frame_len_start * (2 ** i)
            hop_length = frame_length // 2
            dic = {"rec": scipy.signal.windows.boxcar(frame_length), "ham": scipy.signal.windows.hamming(frame_length)}
            win = dic[win_type]

            if time_index:
                start_index = int(time_index * self.sr)
                spec = scipy.fft.fft(self.wave[start_index:start_index + frame_length] * win)
                ax[i].plot(np.abs(spec[:frame_length // 2 - 1]))
                ax[i].set_title("{x},M = {m}".format(x=win_type, m=frame_length), y=0.5)
            else:
                spec = librosa.stft(self.wave * 1.0, n_fft=frame_length, hop_length=hop_length, win_length=frame_length,
                                    window=win)
                spec_maglog = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
                librosa.display.specshow(spec_maglog, sr=self.sr, hop_length=hop_length, ax=ax[i])
        if time_index:
            ax[plot_num].plot(self.wave[start_index:start_index + frame_length])
        else:
            ax[plot_num].plot(self.wave)
        # librosa.display.waveshow(self.wave * 1.0, sr=self.sr, max_points=self.sr // 2, ax=ax[plot_num])
        fig.tight_layout()
        fig.show()


if __name__ == '__main__':
    wav_path = "./data/Male_8k.wav"
    speech = SpeechSignal(wav_path)
    # speech.plotter_p(win_type="ham", hop_length=32, plot_num=4, frame_len_start=64)
    speech.plotter_f(win_type="rec", plot_num=4, frame_len_start=64, time_index=0.85)
