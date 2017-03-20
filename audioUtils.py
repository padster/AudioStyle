import scipy.io.wavfile as wavfile
import numpy as np

def fileToSamples(path):
    print ("Reading from %s..." % path)
    rate, samples = wavfile.read(path)
    return rate, samples / 32768.0

def samplesToFile(path, rate, samples):
    print ("Writing to %s..." % path)
    # Normalize first
    samples = samples / np.max(np.abs(samples))
    samples = samples * 32768.0
    samples = samples.astype(np.int16)
    wavfile.write(path, rate, samples)


# Note: these were taking from my own repo, https://github.com/padster/fftSwap

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
