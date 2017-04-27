"""
Code for running some custom analysis on the sound files.
Not for style transfer itself, but for testing out features to use for the transfer
"""

import lasagne
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import skimage.transform
import scipy

import audioUtils
import imageTransfer
import imgUtils
import mfcc
import vggnet
import viz

def crossCorrelate(a, b):
    return scipy.signal.fftconvolve(a, b[::-1])

def crossCorrelateDiff(a):
    cc = crossCorrelate(a, a)
    return np.diff(cc[len(cc)//2:])

def runMFCCAutocorrelations():
    print "Loading files..."
    r1, s1 = audioUtils.loadFirst10s('data/rock17.wav')
    r2, s2 = audioUtils.loadFirst10s('data/reggae07.wav')
    s1 = audioUtils.preprocess(s1, r1)
    s2 = audioUtils.preprocess(s2, r2)
    assert r1 == r2 and len(s1) == len(s2) # should be 661794 = 2 * 3 * 7^2 * 2251 for marsyas

    print "Generating spectrograms..."
    spec1 = audioUtils.toSpectrogram(s1)
    spec2 = audioUtils.toSpectrogram(s2)

    print "Generating mel spectrograms..."
    melSpec1 = audioUtils.toMelSpectrogram(spec1)
    melSpec2 = audioUtils.toMelSpectrogram(spec2)
    print melSpec1.shape
    print melSpec2.shape

    ax = viz.cleanSubplots(2, 1)
    ax[0].set_title('Content MFCC')
    ax[0].matshow(melSpec1, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    ax[1].set_title('Style MFCC')
    ax[1].matshow(melSpec2, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    plt.show()

    (rows, cols) = melSpec1.shape
    ax = viz.cleanSubplots(2, 1, axes=True)
    ax[0].set_title('Content Row Autocorrelations')
    ax[1].set_title('Style Row Autocorrelations')
    ccRows1, ccRows2 = [], []
    for r in range(rows):
        row1, row2 = melSpec1[r, :], melSpec2[r, :]
        ccd1, ccd2 = crossCorrelateDiff(row1), crossCorrelateDiff(row2)
        ccRows1.append(ccd1)
        ccRows2.append(ccd2)
        ax[0].plot(ccd1)
        ax[1].plot(ccd2)
    plt.show()

    ax = viz.cleanSubplots(2, 1, axes=True)
    ax[0].set_title('Content Column Autocorrelations')
    ax[1].set_title('Style Column Autocorrelations')
    ccCols1, ccCols2 = [], []
    for c in range(cols):
        col1, col2 = melSpec1[:, c], melSpec2[:, c]
        ccd1, ccd2 = crossCorrelateDiff(col1), crossCorrelateDiff(col2)
        ccCols1.append(ccd1)
        ccCols2.append(ccd2)
        ax[0].plot(ccd1)
        ax[1].plot(ccd2)
    plt.show()

    ax = viz.cleanSubplots(2, 1, axes=True)
    ax[0].set_title('Row Autocorrelations')
    ax[0].plot(np.mean(ccRows1, axis=0), 'r', label='content')
    ax[0].plot(np.mean(ccRows2, axis=0), 'b', label='style')
    ax[0].legend()
    ax[1].set_title('Column Autocorrelations')
    ax[1].plot(np.mean(ccCols1, axis=0), 'r', label='content')
    ax[1].plot(np.mean(ccCols2, axis=0), 'b', label='style')
    ax[1].legend()
    plt.show()

def runVggAutocorrelations(layers, showRows=True):
    print "Loading files..."
    r1, s1 = audioUtils.loadFirst10s('data/rock17.wav')
    r2, s2 = audioUtils.loadFirst10s('data/reggae07.wav')
    s1 = audioUtils.preprocess(s1, r1)
    s2 = audioUtils.preprocess(s2, r2)
    assert r1 == r2 and len(s1) == len(s2) # should be 661794 = 2 * 3 * 7^2 * 2251 for marsyas

    print "Generating spectrograms..."
    spec1 = audioUtils.toSpectrogram(s1)
    spec2 = audioUtils.toSpectrogram(s2)

    print "Generating mel spectrograms..."
    melSpec1 = audioUtils.toMelSpectrogram(spec1)
    melSpec2 = audioUtils.toMelSpectrogram(spec2)

    print "Preprocessing mel spectrogram images"
    mx1, mn1 = np.max(melSpec1), np.min(melSpec1)
    mx2, mn2 = np.max(melSpec2), np.min(melSpec2)
    fft1Img = ((melSpec1 - mn1) / (mx1 - mn1) * 256).astype('uint8')
    fft2Img = ((melSpec2 - mn2) / (mx2 - mn2) * 256).astype('uint8')
    # Each will be 96 x 854, no need to downsample
    _, fft1ImgProc = imgUtils.preprocess(fft1Img)
    _, fft2ImgProc = imgUtils.preprocess(fft2Img)

    print "Building VGG.."
    _, _, h, w = fft1ImgProc.shape
    net = vggnet.buildVgg(w, h)

    for layer in layers:
        print 'Compute activations for layer %s' % layer
        input_im_theano = T.tensor4()
        output = lasagne.layers.get_output(net[layer], input_im_theano)
        features1 = theano.shared(output.eval({input_im_theano: fft1ImgProc})).eval()
        features2 = theano.shared(output.eval({input_im_theano: fft2ImgProc})).eval()
        _, depth, rows, cols = features1.shape

        ax = viz.cleanSubplots(2, 1, axes=True)
        if showRows:
            ax[0].set_title('Content Row Autocorrelations for layer ' + layer)
            ax[1].set_title('Style Row Autocorrelations for layer ' + layer)
            for d in range(96 / rows):
                for row in range(rows):
                    row1, row2 = features1[0, d, row, :], features2[0, d, row, :]
                    ccd1, ccd2 = crossCorrelateDiff(row1), crossCorrelateDiff(row2)
                    ax[0].plot(ccd1)
                    ax[1].plot(ccd2)
            plt.show()
        else:
            ax[0].set_title('Content Column Autocorrelations for layer ' + layer)
            ax[1].set_title('Style Column Autocorrelations for layer ' + layer)
            for col in range(0, cols, (cols + 63) // 64):
                col1, col2 = features1[0, 0, :, col], features2[0, 0, :, col]
                ccd1, ccd2 = crossCorrelateDiff(col1), crossCorrelateDiff(col2)
                ax[0].plot(ccd1)
                ax[1].plot(ccd2)
            plt.show()

if __name__ == '__main__':
    runMFCCAutocorrelations()
    #srunVggAutocorrelations(layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'], showRows=True)
