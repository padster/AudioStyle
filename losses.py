import theano.tensor as T

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

def totalVariation(x):
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()
