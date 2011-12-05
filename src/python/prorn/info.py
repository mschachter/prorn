import numpy as np
from scipy.stats import gaussian_kde

from prorn.readout import get_samples

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

def entropy(samps, nbins=64):
    
    (hist, edges) = np.histogramdd(samps, bins=nbins, normed=False)
    hist = np.array(hist, dtype='float')    
    hist /= float(samps.shape[0])
    nzp = hist[hist.nonzero()]
    nzp.sort()

    H = 0.0
    for p in nzp:
        H += p*np.log2(p)
    return -H
