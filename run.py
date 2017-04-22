"""
run.py for 540 project. Runs audio transfer using the image transfer algorithm.
Flags:
 --cpu runs it in CPU mode (with bad quality transfers), default is GPU mode
 --spec runs it by transfering Spectrogram images. Default is transferring MFCC images.
"""

import sys
USE_GPU = "--cpu" not in sys.argv
print "====\nUsing GPU? " + str(USE_GPU) + "\n====\n"

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform

import audioUtils
import imageTransfer
import imgUtils
import mfcc

OUTPUT_FOLDER = "output/"
IMAGE_SZ = 64

# Parameters:
ITERATIONS_GPU_SPEC = 25
ITERATIONS_GPU_MFCC = 40
ITERATIONS_CPU_SPEC = 2
ITERATIONS_CPU_MFCC = 2

# HACK - move elsewhere?
# Subplots helper: hide axes, minimize space between, maximize window
def cleanSubplots(r, c, pad=0.05):
    f, ax = plt.subplots(r, c)
    if r == 1 and c == 1:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    elif r == 1 or c == 1:
        for a in ax:
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
    else:
        for aRow in ax:
            for a in aRow:
                a.get_xaxis().set_visible(False)
                a.get_yaxis().set_visible(False)

    f.subplots_adjust(left=pad, right=1.0-pad, top=1.0-pad, bottom=pad, hspace=pad)
    plt.get_current_fig_manager().window.showMaximized()
    return ax

# HACK - move elsewhere?
# Visualization helper: Show results, or write to file if running on AWS:
def saveOrShow(path):
    plt.savefig(OUTPUT_FOLDER + path)
    if not USE_GPU:
        plt.show()

def centreCrop(img, sz):
    h, w, _ = img.shape
    # TODO - support non-square?
    if h < w:
        img = skimage.transform.resize(img, (sz, w*sz/h), preserve_range=True)
    else:
        img = skimage.transform.resize(img, (h*sz/w, sz), preserve_range=True)
    # Central crop
    h, w, _ = img.shape
    # return img[h//2 - sz//2 : h//2 + sz//2, w//2 - sz//2 : w//2 + sz//2]
    return img[0:sz, 0:sz+16]

def runImageTransferTest():
    print 'content image:'
    photo = plt.imread('data/Tuebingen_Neckarfront.jpg')
    photo = centreCrop(photo, IMAGE_SZ)
    rawim, photo = imgUtils.preprocess(photo)
    plt.imshow(rawim)
    plt.show()

    print 'style image:'
    style = plt.imread('data/1920px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')
    style = centreCrop(style, IMAGE_SZ)
    rawim, style = imgUtils.preprocess(style)
    plt.imshow(rawim)
    plt.show()

    partials = imageTransfer.transfer(photo, style)

    # plt.figure(figsize=(12,12))
    for i in range(len(partials)):
        plt.subplot(3, 3, i+1)
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        plt.imshow(imgUtils.deprocess(partials[i]))
    plt.tight_layout()
    plt.show()

def audioTransferSpec():
    print "Loading files..."
    r1, s1 = audioUtils.loadFirst10s('data/rock17.wav')
    r2, s2 = audioUtils.loadFirst10s('data/reggae07.wav')
    s1 = audioUtils.preprocess(s1, r1)
    s2 = audioUtils.preprocess(s2, r2)
    assert r1 == r2 and len(s1) == len(s2) # should be 220500 = 2^2 * 3^2 * 5^3 * 7^2 for first10s

    print "Generating spectrograms..."
    spec1 = audioUtils.toSpectrogram(s1)
    spec2 = audioUtils.toSpectrogram(s2)

    print "Preprocessing spectrogram images"
    mx1, mn1 = np.max(spec1), np.min(spec1)
    mx2, mn2 = np.max(spec2), np.min(spec2)
    fft1Img = ((spec1 - mn1) / (mx1 - mn1) * 256).astype('uint8')
    fft2Img = ((spec1 - mn2) / (mx2 - mn2) * 256).astype('uint8')
    # Each will be 1024 x 5168, downsample to
    w, h = fft1Img.shape
    ZOOM = 4 # Should be a factor of gcd(1024, 1712) = 16
    fft1Img = skimage.transform.rescale(fft1Img, 1.0/ZOOM, order=3, preserve_range=True)
    fft2Img = skimage.transform.rescale(fft2Img, 1.0/ZOOM, order=3, preserve_range=True)
    _, fft1ImgProc = imgUtils.preprocess(fft1Img)
    _, fft2ImgProc = imgUtils.preprocess(fft2Img)
    print fft2ImgProc.shape

    print "Transferring style from one spectrogram onto the other..."
    ITER = ITERATIONS_GPU_SPEC if USE_GPU else ITERATIONS_CPU_SPEC
    partials = imageTransfer.transfer(fft1ImgProc, fft2ImgProc, iterations=ITER)
    # partials = [fft1ImgProc]
    specOut = imgUtils.deprocess(partials[-1])
    specOut = specOut[:, :, 0] # hack - can only use one channel
    specOut = mn1 + (mx1 - mn1) * specOut / 256.0
    # resize back up:
    specOut = skimage.transform.rescale(specOut, ZOOM, order=3, preserve_range=True)

    ax = cleanSubplots(3, 1)
    ax[0].set_title('Content Spec')
    ax[0].matshow(spec1.T, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    ax[1].set_title('Style Spec')
    ax[1].matshow(spec2.T, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    ax[2].set_title('Result Spec')
    ax[2].matshow(specOut.T, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    saveOrShow("specOut.png")

    print "Inverting spectrogram back to samples..."
    outSamples = audioUtils.fromSpectrogram(specOut)
    audioUtils.samplesToFile(OUTPUT_FOLDER + 'specOut.wav', r1, outSamples)


# NOTE: Doesn't work well even without style transfer yet...need to debug.
def audioTransferMFCC():
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
    print "MEL SIZE:"
    print melSpec1.shape

    if not USE_GPU:
        ax = cleanSubplots(2, 1)
        ax[0].set_title('Content MFCC')
        ax[0].matshow(melSpec1, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
        ax[1].set_title('Style MFCC')
        ax[1].matshow(melSpec2, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
        plt.show()

    print "Preprocessing spectrogram images"
    mx1, mn1 = np.max(melSpec1), np.min(melSpec1)
    mx2, mn2 = np.max(melSpec2), np.min(melSpec2)
    fft1Img = ((melSpec1 - mn1) / (mx1 - mn1) * 256).astype('uint8')
    fft2Img = ((melSpec2 - mn2) / (mx2 - mn2) * 256).astype('uint8')
    # Each will be 96 x 854, no need to downsample
    _, fft1ImgProc = imgUtils.preprocess(fft1Img)
    _, fft2ImgProc = imgUtils.preprocess(fft2Img)
    print fft2ImgProc.shape

    print "Transferring style from one mfcc grid conto the other..."
    ITER = ITERATIONS_GPU_MFCC if USE_GPU else ITERATIONS_CPU_MFCC
    partials = imageTransfer.transfer(fft1ImgProc, fft2ImgProc, iterations=ITER)
    # partials = [fft1ImgProc]
    melSpecOut = imgUtils.deprocess(partials[-1])
    melSpecOut = melSpecOut[:, :, 0] # hack - can only use one channel
    melSpecOut = mn1 + (mx1 - mn1) * melSpecOut / 256.0

    ax = cleanSubplots(3, 1)
    ax[0].set_title('Content MFCC')
    ax[0].matshow(melSpec1, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    ax[1].set_title('Style MFCC')
    ax[1].matshow(melSpec2, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    ax[2].set_title('Result MFCC')
    ax[2].matshow(melSpecOut, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    saveOrShow("melOut.png")

    print "Inverting mel result back to spectrogram..."
    specOut = audioUtils.fromMelSpectrogram(melSpecOut)
    ax = cleanSubplots(3, 1)
    ax[0].set_title('Content Spec')
    ax[0].matshow(spec1.T, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    ax[1].set_title('Style Spec')
    ax[1].matshow(spec2.T, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    ax[2].set_title('Result Spec')
    ax[2].matshow(specOut.T, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    saveOrShow("specOut.png")

    print "Inverting spectrogram back to samples..."
    sInv = audioUtils.fromSpectrogram(specOut)

    print "Saving..."
    audioUtils.samplesToFile(OUTPUT_FOLDER + 'melOut.wav', r1, sInv)


if __name__ == '__main__':
    # runImageTransferTest()
    if "--spec" in sys.argv:
        audioTransferSpec()
    else:
        audioTransferMFCC()
