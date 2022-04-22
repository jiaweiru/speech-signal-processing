import os
import math
import librosa
import scipy.signal
import scipy.fft
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def get_data():
    root_path = "./data/Speech_8k"

    for i, file in enumerate(os.listdir(root_path)):
        file_path = os.path.join(root_path, file)
        _, wave = scipy.io.wavfile.read(file_path)
        if i == 0:
            data = wave
        else:
            data = np.concatenate((data, wave), axis=0)
    return data


def short_term_energy(waveform, frame_length=256, hop_length=256,  win_type='ham'):
    dic = {"rec": scipy.signal.windows.boxcar(frame_length), "ham": scipy.signal.windows.hamming(frame_length)}
    win = dic[win_type]

    frame = librosa.util.frame(waveform, frame_length=frame_length, hop_length=hop_length, axis=0)
    frame_win = frame * win
    energy = np.mean(frame_win * frame_win, axis=1)
    energy = 10 * np.log10(energy)

    return energy


def vector(seq, k):
    l = seq.shape[0] % k
    if l > 0:
        seq_valid = seq[:-l]
    else:
        seq_valid = seq
    v = seq_valid.reshape(-1, k)
    return v


def Quantization(wave, codebook, k):
    e = short_term_energy(wave)
    v = vector(e, k)
    vq = np.zeros(v.shape)
    for i in range(v.shape[0]):
        q = np.argmin(np.sum((v[i] - codebook) * (v[i] - codebook), axis=1))
        vq[i] = codebook[q]
    v = v.reshape(-1)
    vq = vq.reshape(-1)
    return v, vq


class VQ_LBG:
    def __init__(self, cluster_num, sample_num_all, max_iter=3000, epsilon=0.001):
        self.cluster_num = cluster_num
        self.max_iter = max_iter
        self.sample_num_all = sample_num_all
        self.codebook = None
        self.label_all = None
        self.epsilon = epsilon
        assert (int(math.log2(cluster_num)) == (math.log2(cluster_num))), "The cluster number should be integral power of 2, or split method won't work."

    def generate_init_codebook(self, data):
        self.codebook = (np.mean(data, axis=0)).reshape(1, -1)
        self.label_all = np.zeros((data.shape[0],))

    def compute_D_ave(self, data):
        dist = 0
        for m in range(self.codebook.shape[0]):
            idx = (np.argwhere(self.label_all == m)).reshape(-1,)
            for i in idx:
                dist = dist + np.linalg.norm(data[i] - self.codebook[m])
        D_ave = dist / self.sample_num_all
        return D_ave

    def find_nearest_codebook(self, point):
        min_dist = float('inf')
        label = 0
        for m in range(self.codebook.shape[0]):
            dist = np.linalg.norm(point-self.codebook[m])
            if dist < min_dist:
                min_dist = dist
                label = m
        return label

    def split(self, data):
        for m in range(self.codebook.shape[0]):
            self.codebook[m] = (1 + self.epsilon) * self.codebook[m]
            self.codebook = np.vstack((self.codebook, ((1 - self.epsilon) * self.codebook[m])))

    def update_codebook(self, data):
        for m in range(self.codebook.shape[0]):
            idx = (np.argwhere(self.label_all == m)).reshape(-1, )
            self.codebook[m] = np.mean(data[idx], axis=0)

    def fit(self, data):
        self.generate_init_codebook(data)
        global D_ave
        self.split(data)
        for i in range(int(math.log2(self.cluster_num))):
            for it in range(self.max_iter):
                D_ave = self.compute_D_ave(data)
                for id in range(self.sample_num_all):
                    label = self.find_nearest_codebook(data[id])
                    self.label_all[id] = label
                self.update_codebook(data)
                D_ave_new = self.compute_D_ave(data)
                print(f"iter={it}")
                if np.absolute((D_ave - D_ave_new) / D_ave_new) <= self.epsilon:
                    break
            if i is not int(math.log2(self.cluster_num) - 1):
                self.split(data)
            print(f"i={i}")
        return (self.label_all).astype(int), self.codebook


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    K = 1
    n = 2
    N = 2**n
    wav_path = "./data/Speech_8k/S003.wav"
    _, wave_test = scipy.io.wavfile.read(wav_path)
    wave = get_data()
    energy = short_term_energy(wave)
    vec_energy = vector(energy, K)

    vqlbg = VQ_LBG(N, vec_energy.shape[0])
    l, b = vqlbg.fit(vec_energy)

    v, vq = Quantization(wave_test, b, K)

    fig, ax = plt.subplots()
    ax.plot(v)
    ax.plot(vq)
    ax.set_ylabel("dB")
    ax.set_xlabel("Frame index")
    ax.set_title(f"{n}bit")
    print(v.shape)
    print(np.mean((v - vq) * (v - vq)))
    plt.show()
    # ---------------------------------------------------------------------

    # k = 1
    # 2bit:18.286654163440726
    # 3bit:5.1144496399628485
    # 4bit:1.3236989743378842
    # 5bit:0.3357267535923523
    # 6bit:0.07298133942712855
    # 7bit:0.018559147955918467
    # 8bit:0.005505656301693018

    # k = 2
    # 2bit:38.60364133229845
    # 3bit:27.488924874525964
    # 4bit:14.001736494285662
    # 5bit:5.9434025452917885
    # 6bit:3.4634746985149816
    # 7bit:1.637997337781119
    # 8bit:1.1800045286073193

    # k = 3
    # 2bit:53.0592215126667
    # 3bit:36.16585378429846
    # 4bit:20.38372365969113
    # 5bit:15.007615613532304
    # 6bit:7.973363596704419
    # 7bit:4.860936309320153
    # 8bit:3.6135303018521014

    #
    # ---------------------------------------------------------------------
    # k1 = [18.286654163440726, 5.1144496399628485, 1.3236989743378842, 0.3357267535923523, 0.07298133942712855, 0.018559147955918467, 0.005505656301693018]
    # k2 = [38.60364133229845, 27.488924874525964, 14.001736494285662, 5.9434025452917885, 3.4634746985149816, 1.637997337781119, 1.1800045286073193]
    # k3 = [53.0592215126667, 36.16585378429846, 20.38372365969113, 15.007615613532304, 7.973363596704419, 4.860936309320153, 3.6135303018521014]
    # x = [2, 3, 4, 5, 6, 7, 8]
    # fig, ax = plt.subplots()
    # ax.plot(x, k1, label="k=1")
    # ax.plot(x, k2, label="k=2")
    # ax.plot(x, k3, label="k=3")
    # ax.legend()
    # plt.show()
    # ---------------------------------------------------------------------
