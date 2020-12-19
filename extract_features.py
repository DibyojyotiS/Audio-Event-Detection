import librosa
import numpy as np
import os
import multiprocessing as mp
# from scipy.io import savemat

audio_dir = "#shared_train/audio_train"
save_dir  = "cellar/spectograms"

def wav2feat(wavfile):
    '''
    Input: audio wav file name
    Output: Magnitude spectrogram
    '''
    x, Fs = librosa.load(wavfile, sr=44100, mono=True) 
    hop = int(0.01 * Fs) # 10ms
    win = int(0.02 * Fs) # 20ms
    X = librosa.stft(x, n_fft=1024, hop_length=hop, win_length=win, window='hann', center=True, pad_mode='reflect')
    return np.abs(X)


def save(f):
    name = f.split('.')[0]
    spec = wav2feat(f"{audio_dir}/{f}")
    np.save(f"{save_dir}/{name}.npy", spec) # non-compressed numpy version
    # np.savez_compressed(f"{save_dir}/{name}.npy",spectogram= spec) # compressed numpy version
    # mdict = {'spectogram': spec}
    # savemat(f"{save_dir}/{name}.mat", mdict, do_compression=True) # compressed .mat version


if __name__ == "__main__":
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    pool = mp.Pool(processes=5)
    mp.freeze_support()
    pool.map(save, os.listdir(audio_dir))
