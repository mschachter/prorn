import os
import csv
import operator

import numpy as np
import scipy

import h5py

from prorn.readout import train_readout_nn
from prorn.info import entropy_ratio, mutual_information, fisher_memory_matrix
from prorn.network import EchoStateNetwork
from prorn.sim import run_sim
from prorn.spectra import compute_pseudospectra
from prorn.convexhull import convex_hull

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
        net = EchoStateNetwork()
        net.from_hdf5(net_grp)
        net.compile()
        W = net.W
        Win = net.Win
        input_node = Win.argmax()
        lperf = float(net_grp['performance']['standard'][()])
        (evals, evecs) = np.linalg.eig(W)
        
        pdata = PerformanceData()
        pdata.file_name = top_file
        pdata.net = net
        pdata.net_key = net_key
        pdata.input_node = input_node
        pdata.logit_perf = lperf
        pdata.W = W
        pdata.Win = Win.squeeze()
        pdata.eigen_values = evals
        perfs.append(pdata)
    
    f.close()
    
    perfs.sort(key=operator.attrgetter('logit_perf'), reverse=True)
    
    return perfs

def compute_state_rank(perf_data, stimset, stim_class, svd_eps=1e-10):
    net = perf_data.net
    net.noise_std = 0.0
    
    states = []
    
    for md5 in stimset.class_to_md5[stim_class]:
        stim = stimset.all_stims[md5]
        net_sims = run_sim(net, {md5:stim},
                           burn_time=100,
                           pre_stim_time = -1,
                           post_stim_time=1,
                           num_trials=1)
        
        sim = net_sims[md5]
        avg_resp = sim.responses[0, :, :].squeeze()
        states.append(avg_resp)
    
    #X = np.transpose(np.array(states))
    X = np.array(states)
    (U, S, V) = np.linalg.svd(X)
    print S
    rank = np.sum(S > svd_eps)
    
    return rank

def compute_net_metrics(pdata, output_file):
    
    npts = 40
    
    data = []
    
    for p in pdata:
        J = fisher_memory_matrix(p.net.W, p.net.Win, use_dlyap=False, npts=npts)
        
        feature_names = []
        features = []
        
        #compute weighted jtot
        fmc = np.diag(J)
        fmc[fmc < 0.0] = 0.0
        indx = np.arange(len(fmc)) + 1.0
        w = np.log2(indx)
        wjtot = (w * fmc).sum()
        feature_names.append('wjtot')
        features.append(wjtot)
        
        #compute upper-triangular abs sum
        jt_vals = np.abs(J[np.triu_indices(len(J))]) 
        fmm_sum = jt_vals.sum()
        feature_names.append('fmm_sum')
        features.append(fmm_sum)
        
        #compute arc length of convex hull around pseudospectra
        bounds=[-3, 3, -3, 3]
        for eps in [0.5, 0.1]:
            (X, Y, Z, smin) = compute_pseudospectra(p.net.W, bounds=bounds, npts=75, invert=False)
            arclen = np.NAN
            if np.sum(smin < eps) > 0:
                arclen = 0.0                
                xvals = X[smin < eps].ravel()
                yvals = Y[smin < eps].ravel()
                pnts = zip(xvals, yvals)
                ch = np.array(convex_hull(pnts))
                
                for m in range(ch.shape[0]):
                    if m == 0:
                        p1 = ch[-1, :]
                    else:
                        p1 = ch[m-1, :]
                    p2 = ch[m, :]
                    arclen += np.linalg.norm(p2 - p1)
            feature_names.append('arclen_%0.2f' % eps)
            features.append(arclen)
        
        #compute eigenvalues/schur decomposition
        (T, U, sdim) = scipy.linalg.schur(p.net.W, 'complex', sort='rhp')
        
        evals = np.diag(T)
        for k,ev in enumerate(evals):
            feature_names.append('ev%d_real' % k)
            features.append(ev.real)
            feature_names.append('ev%d_imag' % k)
            features.append(ev.imag)
        
        for i in range(p.net.W.shape[0]):
            for j in range(i):
                if i != j:
                    od = T[j, i]
                    feature_names.append('od%d%d_real' % (j, i))
                    features.append(od.real)
                    feature_names.append('od%d%d_imag' % (j, i))
                    features.append(od.imag)
        
        feature_names.append('perf')
        features.append(p.logit_perf)
        
        data.append(features)
    
    f = open(output_file, 'w')
    f.write('%s\n' % ','.join(feature_names))
    for row in data:
        fstr = ['%0.12f' % x for x in row]
        f.write('%s\n' % ','.join(fstr))
    f.close()
        