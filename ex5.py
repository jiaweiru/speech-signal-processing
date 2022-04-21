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
        if i == 10:
            break
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


class VQ_LBG:
    """
    T={X1,X2,...,XM}=data
    Xm={x(m,1),x(m,2)}
    cluster_num = (final)N
    Codebook: C={c1,c2,...,cN}
    Codewords: Cn={c(n,1),c(n,2)}
    Coding region: P={S1, S2,...,SN}
    Q(Xm)=cn
    D_ave = 1/Mk * norm(Xm-Q(Xm))
    """
    def __init__(self, cluster_num, sample_num_all, max_iter=300, epsilon=0.01):
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
            idx = (np.argwhere(self.label_all == m)).reshape(-1,)
            self.codebook[m] = (1 + self.epsilon) * np.mean(data[idx], axis=0)
            self.codebook = np.vstack((self.codebook, ((1 - self.epsilon) * np.mean(data[idx], axis=0))))

    def update_codebook(self, data):
        for m in range(self.codebook.shape[0]):
            idx = (np.argwhere(self.label_all == m)).reshape(-1, )
            self.codebook[m] = np.mean(data[idx], axis=0)

    def fit(self, data):
        self.generate_init_codebook(data)
        global D_ave
        self.split(data)
        for i in range(int(math.log2(self.cluster_num))):
            # Outer loop, split the codebook each iter
            D_ave = self.compute_D_ave(data)
            for it in range(self.max_iter):
                # Inner loop, minimize the D_average
                for id in range(self.sample_num_all):
                    label = self.find_nearest_codebook(data[id])
                    self.label_all[id] = label
                self.update_codebook(data)
                D_ave_new = self.compute_D_ave(data)
                if (D_ave - D_ave_new) / D_ave <= self.epsilon:
                    D_ave = D_ave_new
                    break
            if i is not int(math.log2(self.cluster_num) - 1):
                self.split(data)
        return (self.label_all).astype(int), self.codebook


if __name__ == "__main__":
    wav_path = "./data/Speech_8k/S044.wav"
    _, wave_test = scipy.io.wavfile.read(wav_path)
    wave = get_data()
    energy = short_term_energy(wave)
    vec_energy = vector(energy, 1)
    vqlbg = VQ_LBG(16, vec_energy.shape[0],)
    l ,b = vqlbg.fit(vec_energy)
    print(l)

    energy_test = short_term_energy(wave_test)
    vec_test = vector(energy_test, 1)
    vec_test.reshape(-1)

    fig, ax = plt.subplots()
    print(b)


