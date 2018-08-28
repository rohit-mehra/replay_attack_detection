import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.lib.stride_tricks import as_strided

import soundfile
import librosa
from char_map import char_map, index_map  # only for speech to text


####################################################################################

train_path = os.path.join(os.path.abspath(os.pardir), "data/asvspoof2017/ASVspoof2017_V2_train/")
val_path = os.path.join(os.path.abspath(os.pardir), "data/asvspoof2017/ASVspoof2017_V2_dev/")
test_path = os.path.join(os.path.abspath(os.pardir), "data/asvspoof2017/ASVspoof2017_V2_eval/")


protocol_header = ['wav_id', 'label', 'speaker_id', 'phrase_id', 'env_id', 'pb_device_id', 'rec_device_id']
protocol_path = os.path.join(os.path.abspath(os.pardir), "data/asvspoof2017/protocol_V2/")

train_protocol = os.path.join(os.path.abspath(os.pardir), "data/asvspoof2017/protocol_V2/ASVspoof2017_V2_train.trn.txt")
val_protocol = os.path.join(os.path.abspath(os.pardir), "data/asvspoof2017/protocol_V2/ASVspoof2017_V2_dev.trl.txt")
test_protocol = os.path.join(os.path.abspath(os.pardir), "data/asvspoof2017/protocol_V2/ASVspoof2017_V2_eval.trl.txt")

phrase_ids = {'S01': 'My voice is my password',
              'S02': 'OK Google',
              'S03': 'Only lawyers love millionaires',
              'S04': 'Artificial intelligence is for real',
              'S05': 'Birthday parties have cupcakes and ice cream',
              'S06': 'Actions speak louder than words',
              'S07': 'There is no such thing as a free lunch',
              'S08': 'A watched pot never boils',
              'S09': 'Jealousy has twenty-twenty vision',
              'S10': 'Necessity is the mother of invention'}

####################################################################################


def plot_raw_audio(raw_audio):
    """Plot the raw audio signal"""

    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    steps = len(raw_audio)
    ax.plot(np.linspace(1, steps, steps), raw_audio)
    plt.title('Audio Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def plot_spectrogram_feature(spectrogram_feature):
    """Plot the Spectrogram"""

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(spectrogram_feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized Spectrogram')
    plt.ylabel('Time')
    plt.xlabel('Frequency')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def plot_mfcc_feature(vis_mfcc_feature):
    """Plot the MFCC feature"""

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(vis_mfcc_feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized MFCC')
    plt.ylabel('Time')
    plt.xlabel('MFCC Coefficient')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_xticks(np.arange(0, 13, 2), minor=False)
    plt.show()

####################################################################################

def feat_padding(feat, width=400, flat=True):
    dim = feat.shape[0]

    padding = width - feat.shape[1] % width
    if padding < 200:
        feat = np.pad(feat, [[0, 0], [0, padding]], mode='edge')
    if feat.shape[1] % width:
        feat = feat[:, :-(feat.shape[1]%width)]
    
    tmp_feat = feat.T.reshape(width, dim)
    
    if not flat:
        tmp_feat = tmp_feat.T.reshape(-1, width, dim)

    return tmp_feat

def calc_feat_dim(window, max_freq):
    return int(0.001 * window * max_freq) + 1


def trim_silence(audio, threshold=0.1, frame_length=2048):
    
    if audio.size < frame_length:
        frame_length = audio.size
        
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x

    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs


def spectrogram_from_file(filename, step=10, window=20, max_freq=None, eps=1e-14, trim=False):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    with soundfile.SoundFile(filename) as sound_file:
        
        
        y = sound_file.read(dtype='float32')

        if trim:
            audio = trim_silence(y)
        else:
            audio = y

        if audio.size == 0:
            audio = y
            
        sample_rate = sound_file.samplerate
        
        if audio.ndim >= 2:
            audio = np.mean(audio, 1)
        
        if max_freq is None:
            max_freq = sample_rate / 2
        
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        if step > window:
            raise ValueError("step size must not be greater than window size")
        
        hop_length = int(0.001 * step * sample_rate)
        
        fft_length = int(0.001 * window * sample_rate)
        
        pxx, freqs = spectrogram(
            audio, fft_length=fft_length, sample_rate=sample_rate,
            hop_length=hop_length)
        ind = np.where(freqs <= max_freq)[0][-1] + 1

    return np.transpose(np.log(pxx[:ind, :] + eps))


def extract_fft(filename, hop_length=20, n_fft=798):
    
    y, _ = librosa.load(filename, sr=16000)
    
    hop_length = int(0.001 * 10 * 16000)
    
    S, _ = librosa.core.spectrum._spectrogram(y, hop_length=hop_length, n_fft=n_fft, power=2)
    return feat_padding(S)



def cqt_from_file(filename, step=10, max_freq=None, trim=False):
    
    sample_rate = 16000
    
    with soundfile.SoundFile(filename) as sound_file:
        
        n_cqt = 13
        
        y, sr = librosa.load(filename, sr=sample_rate)

        if trim:
            audio = trim_silence(y)
        else:
            audio = y

        if audio.size == 0:
            audio = y
            
        sample_rate = sound_file.samplerate
        
        if audio.ndim >= 2:
            audio = np.mean(audio, 1)
        
        if max_freq is None:
            max_freq = sample_rate / 2
        
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        
        hop_length = int(0.001 * step * sample_rate)
        
        f_min = max_freq / 2**9
        cqt = librosa.feature.chroma_cqt(audio, sample_rate, hop_length=hop_length, fmin=f_min, n_chroma=n_cqt, n_octaves=5)
        
        # Unable to vstack cqt ValueError: all the input array dimensions except for the concatenation axis must match exactly
        return cqt

####################################################################################


def text_to_int_sequence(text):
    """ Convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence


def int_sequence_to_text(int_sequence):
    """ Convert an integer sequence to text """
    text = []
    for c in int_sequence:
        ch = index_map[c]
        text.append(ch)
    return text


if __name__ == "__main__":

    print(os.path.isdir(train_path))
    print(os.path.isdir(val_path))
    print(os.path.isdir(test_path))

    print(os.path.isdir(protocol_path))

    print(os.path.isfile(train_protocol))
    print(os.path.isfile(val_protocol))
    print(os.path.isfile(test_protocol))
