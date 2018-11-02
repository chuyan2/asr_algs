import numpy as np
import librosa
import scipy
from data_loader import load_audio

def test(y):
    n_fft = int(16000 * 0.02)
    win_length = n_fft
    hop_length = int(16000 * 0.01)
    y=y[:1000]
    print('y',len(y))
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,center=False,
                     win_length=win_length,window=scipy.signal.hamming)
    spect, phase = librosa.magphase(D)
    spect = np.log1p(spect)
    print(spect.size)
    print(spect.shape)

test(load_audio('10c1.wav'))
