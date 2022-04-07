import scipy
from scipy import fft, io, signal
import numpy as np
import matplotlib.pyplot as plt
import librosa


def plot_tf(sig, title=None):
    length = sig.shape[0]
    freq = scipy.fft.fft(sig)
    freq_mag = np.abs(freq)
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(sig)
    ax[0].set_title("time domain")
    # ax[0].set_xlim([0, length])

    x = np.arange(length) / length * 8000
    ax[1].plot(x[:(length // 2 + 1)], 20 * np.log10(freq_mag[:(length // 2 + 1)]))
    ax[1].set_title("frequency domain")
    # ax[1].set_xlim([0, length])
    # sharex=ax[0]
    fig.suptitle(title, x=0.9)
    fig.tight_layout()
    plt.show()


def get_cl(frame):  # signal of a frame
    length = frame.shape[0]
    cl = np.min([np.max(frame[:length // 3]), np.max(frame[-length // 3:])])
    cl = cl * 0.68
    return cl


def center_clip(frame, cl):
    f_cc = np.zeros(frame.shape[0])
    for i, x in enumerate(frame):
        if x > cl:
            f_cc[i] = x - cl
        elif x <= -cl:
            f_cc[i] = x + cl
        elif -cl < x <= cl:
            f_cc[i] = 0
    return f_cc


class SpeechSignal:
    def __init__(self, path):
        self.sr, self.wave = scipy.io.wavfile.read(path)
        print("Data loading completed")

    def preproc(self, frame):
        # frame = scipy.signal.lfilter([0.008233, -0.004879, 0.007632, 0.007632, -0.004879, 0.008233],
        #                              [1., -3.6868, 5.8926, -5.0085, 2.2518, -0.4271],
        #                              frame)
        frame = frame - np.average(frame)
        frame = scipy.signal.filtfilt([0.008233, -0.004879, 0.007632, 0.007632, -0.004879, 0.008233],
                                      [1., -3.6868, 5.8926, -5.0085, 2.2518, -0.4271],
                                      frame)
        elp = np.mean(10 * np.log10(np.sum(frame*frame)))
        center_clip(frame, get_cl(frame))
        return frame, elp

    def pitch_detection_frame(self, frame, elp):
        if elp < 55:
            return 0
        cor = np.correlate(frame, frame, "full")
        cor = cor[-frame.shape[0]:]
        # for i in range(cor.shape[0]):
        #     cor[i] = (cor[i] / (cor.shape[0] - i))
        #     if i > 40:
        #         cor[i] = cor[i] * 0.85
        #     if i > 80:
        #         cor[i] = cor[i] * 0.85
        max_index = np.argmax(cor[16:134])
        max_index = max_index + 16
        if cor[max_index] < 0.25 * cor[0]:
            return 0
        return max_index

    def pitch_detection(self):
        frames = librosa.util.frame(self.wave, frame_length=256, hop_length=128, axis=0)
        p = np.zeros(frames.shape[0])
        p_med = np.zeros(frames.shape[0])
        for i in range(frames.shape[0]):
            p[i] = self.pitch_detection_frame(self.preproc(frames[i, :])[0], self.preproc(frames[i, :])[1])

        p_med[0] = p[0]
        for i in range(1, p.shape[0]-1):
            p_med[i] = np.median(p[i-1:i+1])
        p_med[p.shape[0]-1] = p[p.shape[0]-1]

        fig, ax = plt.subplots(nrows=2)
        ax[0].plot(self.wave)
        ax[0].set_xlim([0, self.wave.shape[0]])
        ax[1].plot(p_med)
        ax[1].set_xlim([0, p.shape[0]])
        ax[1].set_ylim([0, 100])
        plt.show()

    def voiced(self, start):
        length = 256
        frame = self.wave[start:start + length]
        fig, ax = plt.subplots(nrows=3)
        frame_lp = scipy.signal.filtfilt([0.008233, -0.004879, 0.007632, 0.007632, -0.004879, 0.008233],
                                         [1., -3.6868, 5.8926, -5.0085, 2.2518, -0.4271],
                                         frame)
        frame_cc = center_clip(frame_lp, get_cl(frame_lp))
        freq = np.abs(scipy.fft.fft(frame))
        freq_lp = np.abs(scipy.fft.fft(frame_lp))
        freq_cc = np.abs(scipy.fft.fft(frame_cc))
        ax[0].plot(frame)
        ax[0].set_xlim([0, length])
        # ax[0].set_ylim([-2000, 2000])
        ax[1].plot(frame_lp)
        ax[1].set_xlim([0, length])
        # ax[1].set_ylim([-2000, 2000])
        ax[2].plot(frame_cc)
        ax[2].set_xlim([0, length])
        # ax[2].set_ylim([-2000, 2000])
        plt.show()
        fig, ax = plt.subplots(nrows=3)
        x = np.arange(length) / length * 8000
        ax[0].plot(x[:(length // 2 + 1)], 20 * np.log10(freq[:(length // 2 + 1)]))
        ax[0].set_xlim([0, self.sr//2])
        ax[0].set_xlabel("Hz")
        ax[0].set_ylabel("dB")
        ax[1].plot(x[:(length // 2 + 1)], 20 * np.log10(freq_lp[:(length // 2 + 1)]))
        ax[1].set_xlim([0, self.sr//2])
        ax[1].set_xlabel("Hz")
        ax[1].set_ylabel("dB")
        ax[2].plot(x[:(length // 2 + 1)], 20 * np.log10(freq_cc[:(length // 2 + 1)]))
        ax[2].set_xlim([0, self.sr//2])
        ax[2].set_xlabel("Hz")
        ax[2].set_ylabel("dB")
        plt.show()
        return frame


if __name__ == '__main__':
    wav_path = "./data/Female_8k.wav"
    wav_path2 = "./data/Male_8k.wav"
    speech = SpeechSignal(wav_path)
    speech2 = SpeechSignal(wav_path2)
    # speech.voiced(15200)
    # speech2.voiced(15800)
    speech.pitch_detection()
    speech2.pitch_detection()
