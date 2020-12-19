import os
import librosa
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
def single_bar_plot(x, h, plot_dir, title):
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    fig = plt.figure(figsize=[16,5]); ax = fig.add_subplot()
    ax.bar(x, h)
    ax.set_title(title)
    plt.savefig(f"{plot_dir}/{title}.png")
    plt.show()

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


def extract_SIF(raw_spec, W=13, frequency_stride=10):
    """
    raw_spec: spectogram of shape (Fbins, time)
    Fbins: number of freq bins
    returns: array of shape (F, time)
    """
    # frequency downsampling into F bins 
    L = raw_spec.shape[0]
    F = (L-W)//frequency_stride + 1
    SIF = np.zeros((F, raw_spec.shape[1]))
    for i in range(F):
        freq_window = raw_spec[i*frequency_stride:i*frequency_stride+W]
        SIF[i][:] = np.mean(freq_window, 0)    
    # denoise
    SIF_dn = SIF - np.min(SIF, axis=0)
    # augment SIF_dn append per frame time tomain energy
    energy_shorttime = np.sum(SIF_dn, axis=0)
    SIF_aug = np.concatenate(
        [SIF_dn, np.expand_dims(energy_shorttime, 0)], axis= 0)

    return SIF_aug


def extract_mbe(spec, sr=44100, n_fft=1024, n_mels=40):
    # log mel band energies
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    return np.log(np.dot(mel_basis, spec) + 1e-8)


def extract_melspec(spec, n_mels = 64):
    D = spec**2
    S = librosa.feature.melspectrogram(S=D, sr=44100, n_mels= n_mels)
    return S


def categorical_focal_loss(y_true, y_pred, gamma, class_weights):
    y_true_labels = tf.argmax(y_true, axis= 1)
    y_pred = tf.clip_by_value(y_pred, 1E-6, 1 - 1E-6)
    total_loss = []
    for class_label in class_weights.keys():
        class_mask = tf.equal(y_true_labels, class_label)
        y_pred_class = tf.boolean_mask(y_pred, class_mask)
        y_true_class = tf.boolean_mask(y_true, class_mask)
        pt1 = tf.where(tf.equal(y_true_class, 1), y_pred_class, tf.ones_like(y_pred_class))
        class_loss = -tf.reduce_sum(
            class_weights[class_label] * tf.math.pow(1.0 - pt1, gamma) * tf.math.log(pt1)
            )
        total_loss.append(class_loss)
    loss = tf.divide(tf.add_n(total_loss), tf.cast(tf.shape(y_true)[0], y_pred.dtype)) 
    return loss


# the SIF generator for task 1
class task1_SIF_generator():
    def __init__(self, spec_files, labels, batch_size, not_infinite):
        """
        spec_files: filepaths
        labels: one-hot encoding list or np array 
        not_infinite: set True is generator should terminate
        """
        self.count = 0
        self.spec_files = spec_files
        self.maxcount = len(spec_files)
        self.labels = labels
        self.batch_size = batch_size
        self.not_infinite = not_infinite
        self.continue_generation = True

    def make_batch(self):
        self.count = self.count % self.maxcount
        MAX = self.count + self.batch_size

        sifs = []
        batch_labels = []
        max_length = 0
        for i in range(self.count, MAX):

            idx = i % self.maxcount
            batch_labels.append(self.labels[idx])
            raw_spec = np.load(self.spec_files[idx], allow_pickle=True)
            sif = extract_SIF(raw_spec)
            sifs.append(sif)
            max_length = max(max_length, sif.shape[1])

            # terminate for non-infinite
            if self.not_infinite and i >= self.maxcount-1: 
                self.continue_generation = False
                break

        self.count += self.batch_size
        
        for i in range(len(sifs)):
            sif = sifs[i]
            pad_len = max_length - sif.shape[1]
            sifs[i] = np.pad(sif, ((0,0),(0,pad_len))).T

        sifs = np.asarray(sifs, np.float32)
        batch_labels = np.asarray(batch_labels)

        return sifs, batch_labels
        

    def generator(self):
        '''
        files: np Array with file names as byte-strings
        labels: integer labels
        Output: np array of spectrograms, corresponding labels as a list
        '''
        while self.continue_generation:
            yield self.make_batch()

    def reset(self):
        self.count = 0
        self.continue_generation = True



# the SIF generator for task 2
class task2_SIF_generator():
    def __init__(self, spec_files, labels, blank_label, batch_size, not_infinite, feature='sif', pad_mode='constant', pad_labels=False):
        """
        spec_files: filepaths
        labels: list or np array
        not_infinite: set True is generator should terminate
        feature in ['sif', 'mbe', 'melspec']
        """
        self.count = 0
        self.spec_files = spec_files
        self.maxcount = len(spec_files)
        self.labels = labels
        self.blank_label = blank_label
        self.batch_size = batch_size
        self.not_infinite = not_infinite
        self.continue_generation = True
        self.pad_labels = pad_labels
        self.pmode = pad_mode
        self.feature = feature

        if pad_mode == 'constant':
            self.pad_fn = lambda label, pad_len: np.pad(label, [0, pad_len], constant_values= self.blank_label, mode=self.pmode)
        elif pad_mode == 'edge':
            self.pad_fn = lambda label, pad_len: np.pad(label, [0, pad_len], mode=self.pmode)
        else:
            self.pad_fn = lambda label, pad_len: np.pad(label, [0, pad_len], constant_values= self.blank_label, mode=self.pmode)

    def make_batch(self):
        self.count = self.count % self.maxcount
        MAX = self.count + self.batch_size

        sifs = []
        batch_labels = []
        max_spec_length = 0
        max_label_len = 0
        for i in range(self.count, MAX):

            idx = i % self.maxcount
            raw_spec = np.load(self.spec_files[idx], allow_pickle=True)
            max_spec_length = max(max_spec_length, raw_spec.shape[1])
            max_label_len = max(max_label_len, len(self.labels[idx]))

            batch_labels.append(np.asarray(self.labels[idx], np.int32))
            sifs.append(extract_mbe(raw_spec) if self.feature=='mbe' \
                        else extract_SIF(raw_spec) if self.feature=='sif' \
                        else extract_melspec(raw_spec))

            # terminate for non-infinite
            if self.not_infinite and i >= self.maxcount-1: 
                self.continue_generation = False
                break
            
        self.count += self.batch_size
        
        for i in range(len(sifs)):
            sif = sifs[i]
            pad_len = max_spec_length - sif.shape[1]
            sifs[i] = np.pad(sif, ((0,0),(0,pad_len))).T

        if self.pad_labels:
            for i in range(len(batch_labels)):
                label = batch_labels[i]
                pad_len = max_label_len - len(label)
                batch_labels[i] = self.pad_fn(label, pad_len)
            batch_labels = np.asarray(batch_labels, np.int32)

        sifs = np.asarray(sifs, np.float32)

        return {'sif_input':sifs, 'target_labels':batch_labels},
        

    def generator(self):
        '''
        files: np Array with file names as byte-strings
        labels: integer labels
        Output: np array of spectrograms, corresponding labels as a list
        '''
        while self.continue_generation:
            yield self.make_batch()

    def reset(self):
        self.count = 0
        self.continue_generation = True



# display.specshow(librosa.amplitude_to_db(mbe[0]['sif_input'][0].T))
# from scipy.io import wavfile
# y = librosa.istft(spec, 256, 512)
# wavfile.write("a017.wav", 44100, y[0:150000]/max(y[0:5000]))