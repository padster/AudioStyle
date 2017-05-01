import theano
import theano.tensor as T
import theano.tensor.signal.conv as TC

# Loss terms:
def gramMatrix(x):
    x = x.flatten(ndim=3)
    return T.tensordot(x, x, axes=([2], [2]))

def content(P, X, layer):
    return 1./2 * ((X[layer] - P[layer])**2).sum()

def style(A, X, layer):
    s = A[layer].shape
    A = gramMatrix(A[layer])
    G = gramMatrix(X[layer])
    N = s[1]
    M = s[2] * s[3]
    return 1./(4 * N**2 * M**2) * ((G - A)**2).sum()

# Row autocorrelations
def rowAC(idx, div, A, X):
    i1, i2 = idx // div, idx % div
    want = A[0, i1, i2:i2+1, :]
    have = X[0, i1, i2:i2+1, :]
    wantAC = TC.conv2d(want, want[:, ::-1], border_mode='full')
    haveAC = TC.conv2d(have, have[:, ::-1], border_mode='full')
    return ((wantAC - haveAC)**2).sum()

def totalRowAC(A, X, layer):
    aValues, xValues = A, X
    if layer is not None:
        aValues, xValues = A[layer], X[layer]
    s = aValues.shape
    indexes = T.arange(s[1] * s[2])
    components, updates = theano.scan(fn=rowAC,
                                      sequences=indexes,
                                      non_sequences=[s[2], aValues, xValues])
    # Average
    return components.sum() / (s[1] * s[2] * s[3])

# Column autocorrelations
def colAC(idx, div, A, X):
    i1, i2 = idx // div, idx % div
    want = A[0, i1, :, i2:i2+1]
    have = X[0, i1, :, i2:i2+1]
    wantAC = TC.conv2d(want, want[:, ::-1], border_mode='full')
    haveAC = TC.conv2d(have, have[:, ::-1], border_mode='full')
    return ((wantAC - haveAC)**2).sum()

def totalColAC(A, X, layer):
    aValues, xValues = A, X
    if layer is not None:
        aValues, xValues = A[layer], X[layer]
    s = aValues.shape
    indexes = T.arange(s[1] * s[3])
    components, updates = theano.scan(fn=rowAC,
                                      sequences=indexes,
                                      non_sequences=[s[3], aValues, xValues])
    # Average
    return components.sum() / (s[1] * s[2] * s[3])


def totalVariation(x):
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()
