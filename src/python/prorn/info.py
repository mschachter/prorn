import numpy as np
from scipy.stats import gaussian_kde

from pydare import dlyap

from prorn.readout import get_samples

class NDHistogram():
    
    def __init__(self, samps, range=None, nbins=125):
        self.samples = samps
        self.nbins = nbins
        (hist, edges) = np.histogramdd(samps, bins=nbins, normed=False, range=range)
        hist /= float(samps.shape[0])
        self.hist = hist
        self.edges = np.array(edges)
    
    def proportion(self, x):
        """ Returns the proportion of elements inside the bin that x belongs to """
        hindx = []
        for k in range(len(x)):
            indx = 0            
            nzi = (self.edges[k, :] < x[k]).nonzero()[0]            
            if len(nzi) > 0:
                indx = nzi.max()
                if len(nzi) >= self.hist.shape[k]:
                    indx = self.hist.shape[k]-1
            hindx.append(indx) 
        
        return self.hist[ tuple(hindx) ]
        
        
def entropy_ratio(net_file, net_key, readout_window, nbins=27):
    
    (samps, all_stim_classes, Ninternal) = get_samples(net_file, net_key, readout_window)
    nsamps = len(samps)
    
    all_samps = np.zeros([nsamps, Ninternal+1])
    for k,(state,sc) in enumerate(samps):
        all_samps[k, 0:Ninternal] = state
        all_samps[k, -1] = sc
        
    #compute unconditional entropy    
    uc_entropy = entropy(all_samps[:, 0:Ninternal], nbins=nbins)
    
    #compute entropy conditioned on each class
    sc_entropy = {}    
    for sc in all_stim_classes:
        indx = all_samps[:, -1] == sc        
        sc_entropy[sc] = entropy(all_samps[indx, 0:Ninternal], nbins=nbins)
    
    sc_entropies = np.array(sc_entropy.values())
    sc_entropy_all = sc_entropies.sum() / len(all_stim_classes)
    entropy_ratio = sc_entropy_all / uc_entropy

    return (entropy_ratio, uc_entropy, sc_entropy)

def mutual_information(net_file, net_key, readout_window, nbins=350):
    
    (samps, all_stim_classes, Ninternal) = get_samples(net_file, net_key, readout_window)
    nsamps = len(samps)
    
    all_samps = np.array([x[0] for x in samps])
    class_indx = np.array([x[1] for x in samps])
    
    Hc = conditional_entropy(all_samps, class_indx, nbins=nbins)
    H = entropy(all_samps)
    
    return (H - Hc, H)
    

def conditional_entropy(samps, class_index, nbins=350):
    
    rng = []
    for m in range(samps.shape[1]):
        smin = samps[:, m].min()
        smax = samps[:, m].max()
        rng.append( (smin, smax) )
    
    #build histograms to compute probabilities    
    hists = {}
    classes = np.unique(class_index)    
    for c in classes:
        indx = class_index == c
        hists[c] = NDHistogram(samps[indx, :], range=rng, nbins=nbins)
    
    #compute conditional entropy
    H = 0.0
    for c in classes:        
        hist = hists[c]
        nzp = []
        Hc = 0.0
        for k in range(samps.shape[0]):
            x = samps[k, :].squeeze()
            p = hist.proportion(x)
            if p > 0.0:
                nzp.append(p)
        nzp.sort()        
        for p in nzp:
            Hc += p*np.log2(p)
        H += Hc
    H /= len(classes)
    
    return -H


def entropy(samps, nbins=350):
    
    (hist, edges) = np.histogramdd(samps, bins=nbins, normed=False)
    hist = np.array(hist, dtype='float')    
    hist /= float(samps.shape[0])
    nzp = hist[hist.nonzero()]
    nzp.sort()

    H = 0.0
    for p in nzp:
        H += p*np.log2(p)
    return -H


def fisher_memory_matrix(W, v, npts = 15, use_dlyap=False):
    
    J = np.zeros([npts, npts])    
    if use_dlyap:
        Cn = dlyap(W, np.eye(W.shape[0]))
    else:
        Cn = gaussian_covariance_matrix(W)
    
    v = v.reshape([W.shape[0], 1])
    Cninv = np.linalg.inv(Cn)
    Wmat = np.matrix(W)
    for k in range(npts):
        for j in range(npts):
            v1 = Wmat**j * v
            #print 'v1.shape,',v1.shape
            v2 = Cninv * v1
            #print 'v2.shape,',v2.shape            
            v3 = np.transpose(Wmat**k) * v2
            #print 'v3.shape,',v3.shape
            v4 = np.transpose(v) * v3
            #print 'v4.shape,',v4.shape
            J[k, j] = np.array(v4).squeeze()
    return J


def gaussian_covariance_matrix(W, niters=100):    
    sum = 0.0
    Wmat = np.matrix(W)
    for k in range(niters):
        Wk = Wmat**k
        sum += Wk*np.transpose(Wk)
    return sum
    
    
    
    
    
    