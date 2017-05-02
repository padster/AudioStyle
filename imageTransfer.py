# Code from Lasagne Neural style recipe:
# https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb
import sys
ROW_AC_LOSS = "--rowac" in sys.argv
COL_AC_LOSS = "--colac" in sys.argv

import lasagne
import numpy as np
import scipy
import theano
import theano.tensor as T
from lasagne.utils import floatX

# Our code
import imgUtils
import losses
import vggnet


def transfer(photo, style, iterations=9,
             contentCost=0.001, styleCost=0.2e6, varCost=0.1e-7, rowACCost=1.e-9, colACCost=1e-9):
    print "Performing image transfer, with %d iterations" % iterations
    _, _, h, w = photo.shape
    _, _, h2, w2 = style.shape
    print photo.shape
    print style.shape
    assert h == h2 and w == w2

    net = vggnet.buildVgg(w, h)

    # Layers for loss calculation:
    layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    layers = {k: net[k] for k in layers}

    # Precompute layer activations for photo and artwork
    print 'Precompute activations...'
    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
    photo_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                      for k, output in zip(layers.keys(), outputs)}
    style_features = {k: theano.shared(output.eval({input_im_theano: style}))
                      for k, output in zip(layers.keys(), outputs)}

    # Get expressions for layer activations for generated image
    print 'Generating feature expressions'
    generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, h, w))))
    gen_features = lasagne.layers.get_output(layers.values(), generated_image)
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}

    # Define loss function
    lossParts = [
        # content loss
        contentCost * losses.content(photo_features, gen_features, 'conv4_2'),
        # style loss
        styleCost * losses.style(style_features, gen_features, 'conv1_1'),
        styleCost * losses.style(style_features, gen_features, 'conv2_1'),
        styleCost * losses.style(style_features, gen_features, 'conv3_1'),
        styleCost * losses.style(style_features, gen_features, 'conv4_1'),
        styleCost * losses.style(style_features, gen_features, 'conv5_1'),
        # total variation penalty
        varCost * losses.totalVariation(generated_image),
    ]
    if ROW_AC_LOSS:
        lossParts.extend([
            # Autocorrelation:
            rowACCost * losses.totalRowAC(style, generated_image, None),
            # rowACCost * losses.totalRowAC(style_features, gen_features, 'conv1_1'),
            # rowACCost * losses.totalRowAC(style_features, gen_features, 'conv2_1'),
        ])
    if COL_AC_LOSS:
        lossParts.extend([
            # Autocorrelation:
            colACCost * losses.totalColAC(style, generated_image, None),
            # colACCost * losses.totalColAC(style_features, gen_features, 'conv1_1'),
            # colACCost * losses.totalColAC(style_features, gen_features, 'conv2_1'),
        ])
    totalLoss = sum(lossParts)

    # Theano functions to evaluate loss and gradient
    print 'Building gradient...'
    f_loss = theano.function([], totalLoss)
    f_grad = theano.function([], T.grad(totalLoss, generated_image))

    # Initialize with a noise image
    print 'Initializing noisy image...'
    generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, h, w))))
    xAt = generated_image.get_value().astype('float64')
    xs = [xAt]

    # Helper functions to interface with scipy.optimize
    def eval_loss(x0):
        x0 = floatX(x0.reshape((1, 3, h, w)))
        generated_image.set_value(x0)
        return f_loss().astype('float64')
        # Losses should end up in the hundreds, or lower for mfcc

    def eval_grad(x0):
        x0 = floatX(x0.reshape((1, 3, h, w)))
        generated_image.set_value(x0)
        return np.array(f_grad()).flatten().astype('float64')

    # Optimize, saving the result periodically
    print 'Optimizing image to reduce loss....'
    for i in range(iterations - 1):
        print(i+1)
        scipy.optimize.fmin_l_bfgs_b(eval_loss, xAt.flatten(), fprime=eval_grad, maxfun=40, iprint=0)
        xAt = generated_image.get_value().astype('float64')
        xs.append(xAt)
        print f_loss()

    return net, xs
