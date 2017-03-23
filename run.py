import matplotlib.pyplot as plt
import numpy as np
import skimage.transform

import audioUtils
import imageTransfer
import imgUtils
import mfcc

IMAGE_SZ = 64

# HACK - move elsewhere
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

def runAudioTransferTest():
    r1, s1 = audioUtils.fileToSamples('data/marsyas/asWav/rock17.wav')
    r2, s2 = audioUtils.fileToSamples('data/marsyas/asWav/reggae07.wav')
    assert r1 == r2 and len(s1) == len(s2) # should be 661794 = 2 * 3 * 7^2 * 2251 for marsyas

    CHUNK_SIZE = 294
    fft1 = audioUtils.stft(s1, CHUNK_SIZE)
    fft2 = audioUtils.stft(s2, CHUNK_SIZE)

    fft1Img = np.log1p(np.abs(fft1))
    fft2Img = np.log1p(np.abs(fft2))
    # mx = max(np.max(fft1Img), np.max(fft2Img))
    # mn = min(np.min(fft1Img), np.min(fft2Img))
    mx1, mn1 = np.max(fft1Img), np.min(fft1Img)
    mx2, mn2 = np.max(fft2Img), np.min(fft2Img)
    fft1Img = ((fft1Img - mn1) / (mx1 - mn1) * 256).astype('uint8')
    fft2Img = ((fft2Img - mn2) / (mx2 - mn2) * 256).astype('uint8')
    _, fft1ImgProc = imgUtils.preprocess(fft1Img)
    _, fft2ImgProc = imgUtils.preprocess(fft2Img)

    print "Running..."
    partials = imageTransfer.transfer(fft1ImgProc, fft2ImgProc, iterations=4)
    print len(partials)
    result = imgUtils.deprocess(partials[-1])
    result = result[:, :, 0] # hack - can only use one channel
    result = mn1 + (mx1 - mn1) * result / 256
    outPower = np.expm1(result)
    outPhase = np.angle(fft1) # hack - should transfer angle style too
    outSpec = outPower * np.exp(1j * outPhase)
    outSamples = audioUtils.rstft(outSpec, CHUNK_SIZE)
    audioUtils.samplesToFile('out.wav', r1, outSamples)

    ax = cleanSubplots(3, 1)
    ax[0].imshow(fft1Img, cmap='gist_heat_r')
    ax[1].imshow(fft2Img, cmap='gist_heat_r')
    ax[2].imshow(result, cmap='gist_heat_r')
    plt.show()


# NOTE: Doesn't work well even without style transfer yet...need to debug.
def mfccSpectrogram():
    print "Loading files..."
    r1, s1 = audioUtils.fileToSamples('data/marsyas/asWav/rock17.wav')
    r2, s2 = audioUtils.fileToSamples('data/marsyas/asWav/reggae07.wav')
    s1 = audioUtils.preprocess(s1, r1)
    s2 = audioUtils.preprocess(s2, r2)
    assert r1 == r2 and len(s1) == len(s2) # should be 661794 = 2 * 3 * 7^2 * 2251 for marsyas

    print "Generating spectrograms..."
    spec1 = audioUtils.toSpectrogram(s1)
    spec2 = audioUtils.toSpectrogram(s2)
    ax = cleanSubplots(2, 1)
    ax[0].matshow(spec1.T, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    ax[1].matshow(spec2.T, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    plt.show()

    print "Generating mel spectrograms..."
    melSpec1 = audioUtils.toMelSpectrogram(spec1)
    melSpec2 = audioUtils.toMelSpectrogram(spec2)
    ax = cleanSubplots(2, 1)
    ax[0].matshow(melSpec1, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    ax[1].matshow(melSpec2, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    plt.show()

    print "Transferring style from style mel onto content mel..."
    # TODO: transfer style

    print "Inverting mel result back to spectrogram..."
    specInv = audioUtils.fromMelSpectrogram(melSpec1)
    print "Inverting spectrogram back to samples..."
    sInv = audioUtils.fromSpectrogram(specInv)
    ax = cleanSubplots(2, 1)
    ax[0].plot(s1)
    ax[1].plot(sInv)
    plt.show()

    print "Saving..."
    audioUtils.samplesToFile('melOut.wav', r1, sInv)


if __name__ == '__main__':
    # runImageTransferTest()
    # runAudioTransferTest()
    mfccSpectrogram()
