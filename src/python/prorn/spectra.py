import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def compute_pseudospectra(A, bounds, npts, invert=True):
    
    (minr, maxr, mini, maxi) = bounds
    
    raxis = np.linspace(minr, maxr, npts)
    iaxis = np.linspace(mini, maxi, npts)
    
    (X, Y) = np.meshgrid(raxis, iaxis)
    
    Z = np.zeros([npts, npts], dtype='complex')
    smin = np.zeros([npts, npts])
    I = np.eye(len(A))
    for m in range(npts):
        for n in range(npts):
            Z[m, n] = np.complex(X[m, n], Y[m, n])
            R = Z[m, n]*I - A
            (U, S, V) = np.linalg.svd(R)
            smin[m][n] = np.min(S)
    
    if invert:
        smin = smin**-1
    
    return (X, Y, Z, smin)

def plot_pseudospectra(A, bounds=[-1, 1, -1, 1], npts=50, ax=None, colorbar=True, log=True):
    
    (X, Y, Z, smin) = compute_pseudospectra(A, bounds, npts)
    
    fig = plt.gcf()    
    if ax is None:        
        ax = fig.add_subplot(1, 1, 1)
    if log:
        smin = np.log10(smin)
    res = ax.imshow(smin, interpolation='nearest', cmap=cm.jet, extent=bounds)
    if colorbar:
        fig = plt.gcf()
        fig.colorbar(res)

def plot_pseudospectra_contour(A, bounds=[-1, 1, -1, 1], npts=50, ax=None, colorbar=True, log=True, levels=None, invert=False):
    
    (X, Y, Z, smin) = compute_pseudospectra(A, bounds, npts, invert=invert)
    
    fig = plt.gcf()    
    if ax is None:        
        ax = fig.add_subplot(1, 1, 1)
    if log:
        smin = np.log10(smin)
    if levels is None:
        res = ax.contour(X, Y, smin)
        plt.clabel(res, fontsize=9, inline=1)
    else:
        res = ax.contour(X, Y, smin, levels=levels)
        
    
    
