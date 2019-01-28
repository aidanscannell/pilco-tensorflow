import numpy as np


def gTrig(m, v, i, e=None):
    """
    Compute moments of the saturating function $e*sin(x(i))$ and $e*cos(x(i))$, where $x \sim\mathcal N(m,v)$ and 
    $i$ is a (possibly empty) set of $I$ indices. The optional  scaling factor $e$ is a vector of length $I$. 
    Optionally, compute derivatives of the moments.
    Args:
        m: mean vector of Gaussian                                    [ d       ]
        v: covariance matrix                                          [ d  x  d ]
        i: vector of indices of elements to augment                   [ I  x  1 ]
        e: (optional) scale vector; default: 1                        [ I  x  1 ]

    Returns:
        M: output means                                              [ 2I       ]
        V: output covariance matrix                                  [ 2I x  2I ]
        C: inv(v) times input-output covariance                      [ d  x  2I ]
    """
    i = i - 1
    d = m.shape[0]
    I = i.shape[0]

    Ic = 2 * np.arange(0, I) + 1
    Is = Ic - 1
    if e is None:
        e = np.ones([I, 1])

    ee = np.tile(e, (2, 1))
    mi = m[i]
    vi = v[:, i][i, :]
    vii = np.diag(vi).reshape(-1, 1)

    M = np.zeros([d, 1])
    M[Is, :] = e * np.exp(-vii / 2) * np.sin(mi)
    M[Ic, :] = e * np.exp(-vii / 2) * np.cos(mi)

    lq = -(vii + vii.T) / 2
    q = np.exp(lq)

    U1 = (np.exp(lq + vi) - q) * np.sin(mi - mi.T)
    U2 = (np.exp(lq - vi) - q) * np.sin(mi + mi.T)
    U3 = (np.exp(lq + vi) - q) * np.cos(mi - mi.T)
    U4 = (np.exp(lq - vi) - q) * np.cos(mi + mi.T)

    V = np.zeros([d, d])
    V[Is[:, np.newaxis], Is] = U3 - U4
    V[Ic[:, np.newaxis], Ic] = U3 + U4
    V[Is[:, np.newaxis], Ic] = U1 + U2
    V[Ic[:, np.newaxis], Is] = (U1 + U2).T
    V = ee * ee.T * V / 2  # variance

    C = np.zeros([d, 2 * I])
    C[i[:, np.newaxis], Is] = np.diag(M[Ic].flatten())
    C[i[:, np.newaxis], Ic] = np.diag(-M[Is].flatten())

    return M, V, C


#
# m = np.array([[1, 2, 3, 4]]).T
#
# # s = 0.1 * np.ones([4, 4])
# s = np.arange(1, 17).reshape(4, 4)
#
# plant.angi = np.array([1, 4])
# gTrig(m, s, plant.angi)
