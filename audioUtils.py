import scipy.io.wavfile as wavfile
import numpy as np

import mfcc

# Taken from: https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html

### AUDIO PROCESSING CONSTANTS ###
FFT_SIZE = 2048 # window size for the FFT
STEP_SIZE = FFT_SIZE/16 # distance to slide along the window (in time)
SPEC_THRESH = 4 # threshold for spectrograms (lower filters out more noise)
LOW_CUT = 125 # Hz # Low cut for our butter bandpass filter
HIGH_CUT = 10000 # Hz # High cut for our butter bandpass filter
LOG_SPECTROGRAM = True
N_MEL_FREQ = 96 # number of mel frequency channels
SHORTEN_FACTOR = 2 # how much should we compress the x-axis (time)
START_FREQ = 125 # Hz # What frequency to start sampling our melS from
END_FREQ = 8000 # Hz # What frequency to stop sampling our melS from

MEL_FILTER, MEL_FILTER_INV = mfcc.create_mel_filter(
    fft_size=FFT_SIZE, n_freq_components=N_MEL_FREQ, start_freq=START_FREQ, end_freq=END_FREQ)

# File -> (Rate, Samples array)
def fileToSamples(path):
    print ("Reading from %s..." % path)
    rate, samples = wavfile.read(path)
    return rate, samples / 32768.0

def loadFirst10s(path):
    rate, samples = fileToSamples(path)
    return rate, samples[:10*rate]

# (Samples, Rate) -> Preprocessed Samples
def preprocess(samples, rate):
    return mfcc.butter_bandpass_filter(samples, LOW_CUT, HIGH_CUT, rate, order=1)

# 1D Samples -> 2D frequency spectrogram matrix
def toSpectrogram(samples):
    return mfcc.pretty_spectrogram(samples,
        fft_size=FFT_SIZE, step_size=STEP_SIZE, thresh=SPEC_THRESH, log=LOG_SPECTROGRAM)

# 2D frequency spectrogram matrix -> 2D mel spectrogram matrix
def toMelSpectrogram(spectrogram):
    return mfcc.make_mel(spectrogram, MEL_FILTER, shorten_factor=SHORTEN_FACTOR)

# Invert the toMelSpectrogram function
def fromMelSpectrogram(melSpectorgram):
    return mfcc.mel_to_spectrogram(melSpectorgram, MEL_FILTER_INV,
        spec_thresh=SPEC_THRESH, shorten_factor=SHORTEN_FACTOR).T

# Invert the toSpectrogram function
def fromSpectrogram(spectrogram):
    return mfcc.invert_pretty_spectrogram(spectrogram,
        fft_size=FFT_SIZE, step_size=STEP_SIZE, n_iter=20, log=LOG_SPECTROGRAM)



def samplesToFile(path, rate, samples):
    print ("Writing to %s..." % path)
    # Normalize first
    samples = samples / np.max(np.abs(samples))
    samples = samples * 32768.0
    samples = samples.astype(np.int16)
    wavfile.write(path, rate, samples)

# SPECTROGRAM


# MFCC STUFF




# Note: these were taking from my own repo, https://github.com/padster/fftSwap
# TODO - remove, replace with nicer ones.

# Utility for chunked short-time fourier transform, given input and chunk size
def stft(x, sz):
    assert sz % 2 == 0
    assert len(x) % sz == 0
    # Return has usual formatting: rows = frequency (high to low), columns = time
    return np.flipud(np.array([np.fft.rfft(x[i:i + sz]) for i in range(0, len(x), sz)]).T)

# Utility for reversing a chunked spectrogram back into samples.
def rstft(s, sz):
    # required: x == rstft(stft(x, sz), sz)
    assert sz % 2 == 0
    s = np.flipud(s).T
    return np.array([np.fft.irfft(s[i, :]) for i in range(0, s.shape[0])]).flatten()
