import matplotlib.pyplot as plt
import skimage.transform

import imageTransfer
import imgUtils

IMAGE_SZ = 64

def centreCrop(img, sz):
    h, w, _ = img.shape
    # TODO - support non-square?
    if h < w:
        img = skimage.transform.resize(img, (sz, w*sz/h), preserve_range=True)
    else:
        img = skimage.transform.resize(img, (h*sz/w, sz), preserve_range=True)
    # Central crop
    h, w, _ = img.shape
    return img[h//2 - sz//2 : h//2 + sz//2, w//2 - sz//2 : w//2 + sz//2]

if __name__ == '__main__':
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
