import os
import csv
import operator

import numpy as np

import h5py

from prorn.readout import train_readout_nn
from prorn.info import entropy_ratio, mutual_information
from prorn.examples import run_many_nets

class PerformanceData:
    def __init__(self):
        self.file_name = None
        self.net_key = None
        self.W = None
        self.input_node = None
        self.logit_perf = None
        
        
def top100_weight_pca(pdata, rootdir='/home/cheese63/git/prorn/data'):
    
    
    pdata.sort(key=operator.attrgetter('nn_perf'))
    weights = []
    for p in pdata[0:100]:
        fname = os.path.join(rootdir, p.file_name)
        net_key = p.net_key
        f = h5py.File(fname, 'r')
        W = np.array(f[net_key]['W'])
        weights.append(W)        
        f.close()
    
    weights = np.array(weights)
    
    wmean = weights.mean(axis=0).squeeze()
    wstd = weights.std(axis=0).squeeze()
    
    wcov = np.zeros([9, 9])
    wmean_flat = wmean.ravel().squeeze()
    for k in range(100):
        wflat = weights[k, :, :].ravel().squeeze()
        wms = wflat - wmean_flat
        wcov += np.outer(wms, wms)
    wcov /= 100
    
    (evals, evecs) = np.linalg.eig(wcov)
    
    evord = zip(range(9), evals)
    evord.sort(key=operator.itemgetter(1), reverse=True)
    
    eval_ret = [x[1] for x in evord]
    evec_ret = [evecs[:, x[0]].squeeze() for x in evord]
    
    proj = []
    for k in range(100):
        p = []
        w = weights[k, :, :].ravel()
        for m in range(3):
            p.append(np.dot(evec_ret[m].ravel(), w))
        proj.append(p)
    
    proj = np.array(proj)
    
    nn_perfs = np.array([p.nn_perf for p in pdata[0:100]])
    
    return (weights, wcov, eval_ret, evec_ret, proj, nn_perfs)    
    

def compute_mutual_information(net_files, readout_window=(0, 1)):
    
    for net_file in net_files:
        if os.path.exists(net_file):    
            fnet = h5py.File(net_file, 'a')
            net_keys = fnet.keys()
            print 'Computing MI for %d networks in %s...' % (len(net_keys), net_file)    
            
            nbins_to_try = [250, 300, 350]
            
            for net_key in net_keys:
                print '\tComputing MI for %s' % net_key
                mis = []
                Hs = []
                for nbins in nbins_to_try:
                    mi = np.nan
                    H = np.nan                    
                    try:            
                        (mi, H) = mutual_information(net_file, net_key, readout_window=readout_window, nbins=nbins)
                    except:
                        print 'Problem with key %s' % net_key
                    mis.append([nbins, mi])
                    Hs.append([nbins, H])
                fnet[net_key].attrs['mutual_information'] = np.array(mis)
                fnet[net_key].attrs['entropy'] = np.array(Hs)
                
            fnet.close()
    
def get_top_perfs(top_file):
    
    perfs = []
    f = h5py.File(top_file, 'r')
    for net_key in f.keys():
        net_grp = f[net_key]
        W = np.array(net_grp['W'])
        Win = np.array(net_grp['Win'])
        input_node = Win.argmax()
        lperf = float(net_grp['performance']['standard'][()])
        (evals, evecs) = np.linalg.eig(W)
        
        pdata = PerformanceData()
        pdata.file_name = top_file
        pdata.input_node = input_node
        pdata.logit_perf = lperf
        pdata.W = W
        pdata.Win = Win.squeeze()
        pdata.eigen_values = evals
        perfs.append(pdata)
    
    f.close()
    
    perfs.sort(key=operator.attrgetter('logit_perf'), reverse=True)
    
    return perfs
