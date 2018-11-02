import os
import librosa
import numpy as np
import scipy.signal
import torch
import math
import wave

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


def load_audio_wave(path):
    f = wave.open(path,'rb')
    b = f.readframes(f.getnframes())
    v = np.fromstring(b[:len(b)-len(b)%8],np.int32)
    f.close()
    return np.float32(v)

def load_audio_torch(path):
    import torchaudio
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound

load_audio = load_audio_wave

def save_audio(v,save_name='tmp.wav'):
    print('type v',type(v[0]))
    if isinstance(v[0],np.float32):
        v = np.int32(v)
    print(v[-10:])
    f = wave.open(save_name,'wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    f.writeframes(v.tostring())
    f.close()
   


class SpectrogramParser():
    def __init__(self, audio_conf):

        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])

        self.mean = -16
        self.std = 1.6

        self.n_fft = int(self.sample_rate * self.window_size)
        self.hop_length = int(self.sample_rate * self.window_stride)
    def parse_audio(self,audio_path):
        raise NotImplementedError

    def wav_vector2nn_input(self,v):
        # STFT
        D = librosa.stft(v, n_fft=self.n_fft, hop_length=self.hop_length,center=True,
                         win_length=self.n_fft, window=self.window)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)

#        spect.add_(self.mean)
#        spect.div_(self.std)
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)
        return spect

