import numpy as np

from lasagne.utils import floatX

MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

def lastAxisFirst(img):
    return np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)

def firstAxisLast(img):
    return np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)

def forceRGB(img):
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.repeat(im, 3, axis=2)
    assert len(img.shape) == 3
    return img

def preprocess(img):
    img = forceRGB(img)
    h, w, _ = img.shape
    rawimg = np.copy(img).astype('uint8')

    # Convert RGB to BGR, normalize for VGG
    img = lastAxisFirst(img)[::-1, :, :] - MEAN_VALUES
    return rawimg, floatX(img[np.newaxis])


def deprocess(img):
    img = (np.copy(img[0]) + MEAN_VALUES)[::-1]
    img = firstAxisLast(img)
    return np.clip(img, 0, 255).astype('uint8')
