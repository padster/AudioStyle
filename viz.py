import sys
USE_GPU = "--cpu" not in sys.argv

import lasagne
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import theano
import theano.tensor as T

from lasagne.utils import floatX

OUTPUT_FOLDER = "output/"

# Subplots helper: hide axes, minimize space between, maximize window
def cleanSubplots(r, c, pad=0.05, axes=False):
    f, ax = plt.subplots(r, c)
    if not axes:
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

    f.subplots_adjust(left=pad, right=1.0-pad, top=1.0-pad, bottom=pad, hspace=0.2)
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except AttributeError:
        pass # Can't maximize, sorry :(
    return ax

# Visualization helper: Show results, or write to file if running on AWS:
def saveOrShow(path):
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.savefig(OUTPUT_FOLDER + path)
    if not USE_GPU:
        plt.show()

### Autocorrelations
def crossCorrelate(a, b):
    return scipy.signal.fftconvolve(a, b[::-1])

def crossCorrelateDiff(a):
    cc = crossCorrelate(a, a)
    return np.diff(cc[len(cc)//2:])

# Row autocorrelation vizualization
def specRowAC(specImg):
    corrs = []
    for r in range(len(specImg)):
        corrs.append(crossCorrelateDiff(specImg[r]))
    return np.mean(corrs, axis=0)

def activationRowAC(title, layerMap, tfImg):
    print 'Computing row AC for %s...' % title
    result = {}
    ftfImg = floatX(tfImg)
    img = T.tensor4()
    for layer in layerMap.keys():
        output = lasagne.layers.get_output(layerMap[layer], img)
        activations = theano.shared(output.eval({img: ftfImg})).get_value().astype('float64')
        _, d, r, c = activations.shape
        corrs = []
        for dd in range(d):
            for rr in range(r):
                row = activations[0, dd, rr, :]
                corrs.append(crossCorrelateDiff(row))
        result[layer] = np.mean(corrs, axis=0)
    return result

def showRowAutocorrlations(vgg, sContent, sStyle, sResult, tfContent, tfStyle, tfResult):
    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    layerMap = {layer: vgg[layer] for layer in layers}
    contentMap = activationRowAC('Content', layerMap, tfContent)
    styleMap   = activationRowAC('Style'  , layerMap, tfStyle)
    resultMap  = activationRowAC('Result' , layerMap, tfResult)

    SPEC = 'input'
    layers.insert(0, SPEC)
    contentMap[SPEC] = specRowAC(sContent)
    styleMap[SPEC]   = specRowAC(sStyle)
    resultMap[SPEC]  = specRowAC(sResult)

    ax = cleanSubplots(len(layers), 1)
    for l in range(len(layers)):
        layer = layers[l]
        ax[l].set_title('Layer: %s' % layer)
        ax[l].plot(contentMap[layer], 'r')
        ax[l].plot(styleMap[layer], 'b')
        ax[l].plot(resultMap[layer], 'g')
        off = len(contentMap[layer]) // 40 + 1
        minY = np.min([np.min(contentMap[layer][off:]), np.min(styleMap[layer][off:]), np.min(resultMap[layer][off:])])
        maxY = np.max([np.max(contentMap[layer][off:]), np.max(styleMap[layer][off:]), np.max(resultMap[layer][off:])])
        ax[l].set_ylim([minY, maxY])
    saveOrShow('row_ac.png')

# Column autocorrelation vizualization
def specColAC(specImg):
    corrs = []
    for c in range(len(specImg[0])):
        corrs.append(crossCorrelateDiff(specImg[:, c]))
    return np.mean(corrs, axis=0)

def activationColAC(title, layerMap, tfImg):
    print 'Computing col AC for %s...' % title
    result = {}
    ftfImg = floatX(tfImg)
    img = T.tensor4()
    for layer in layerMap.keys():
        output = lasagne.layers.get_output(layerMap[layer], img)
        activations = theano.shared(output.eval({img: ftfImg})).get_value().astype('float64')
        _, d, r, c = activations.shape
        corrs = []
        for dd in range(d):
            for cc in range(c):
                col = activations[0, dd, :, cc]
                corrs.append(crossCorrelateDiff(col))
        result[layer] = np.mean(corrs, axis=0)
    return result

def showColAutocorrlations(vgg, sContent, sStyle, sResult, tfContent, tfStyle, tfResult):
    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    layerMap = {layer: vgg[layer] for layer in layers}
    contentMap = activationColAC('Content', layerMap, tfContent)
    styleMap   = activationColAC('Style'  , layerMap, tfStyle)
    resultMap  = activationColAC('Result' , layerMap, tfResult)

    SPEC = 'input'
    layers.insert(0, SPEC)
    contentMap[SPEC] = specColAC(sContent)
    styleMap[SPEC]   = specColAC(sStyle)
    resultMap[SPEC]  = specColAC(sResult)

    ax = cleanSubplots(len(layers), 1)
    for l in range(len(layers)):
        layer = layers[l]
        ax[l].set_title('Layer: %s' % layer)
        ax[l].plot(contentMap[layer], 'r')
        ax[l].plot(styleMap[layer], 'b')
        ax[l].plot(resultMap[layer], 'g')
        off = len(contentMap[layer]) // 40 + 1
        minY = np.min([np.min(contentMap[layer][off:]), np.min(styleMap[layer][off:]), np.min(resultMap[layer][off:])])
        maxY = np.max([np.max(contentMap[layer][off:]), np.max(styleMap[layer][off:]), np.max(resultMap[layer][off:])])
        ax[l].set_ylim([minY, maxY])
    saveOrShow('col_ac.png')
