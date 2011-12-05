import os

import numpy as np

import h5py

from prorn.readout import train_readout_nn, train_readout_logit, train_readout_svm
from prorn.info import entropy_ratio

def compute_readout_perf_nn(net_file, net_range=None):
    
    fnet = h5py.File(net_file, 'a')
    net_keys = fnet.keys()
    if net_range is not None:
        net_keys = net_keys[net_range[0]:net_range[1]]
    print 'Computing performance for %d networks...' % len(net_keys)    
    
    for net_key in net_keys:
        print '----------- Computing NN performance for %s ---------------' % net_key
        (train_err, test_err, fnn, trainer) = train_readout_nn(net_file, net_key, readout_window=np.arange(0, 1))
        fnet[net_key].attrs['nn_perf'] = test_err
        
    fnet.close()
    
def compute_readout_perf_logit(net_file, net_range=None):
    
    fnet = h5py.File(net_file, 'a')
    net_keys = fnet.keys()
    if net_range is not None:
        net_keys = net_keys[net_range[0]:net_range[1]]
    print 'Computing performance for %d networks...' % len(net_keys)    
    
    for net_key in net_keys:
        print '----------- Computing Logit performance for %s ---------------' % net_key
        (test_loss_avg, test_loss_std) = train_readout_logit(net_file, net_key, readout_window=np.arange(0, 1))
        fnet[net_key].attrs['logit_perf'] = test_loss_avg
        fnet[net_key].attrs['logit_perf_std'] = test_loss_std
        print '%s: %0.3f +/- %0.3f' % (net_key, test_loss_avg, test_loss_std)
        
    fnet.close()
    
def compute_readout_perf_svm(net_file, net_range=None):
    
    fnet = h5py.File(net_file, 'a')
    net_keys = fnet.keys()
    if net_range is not None:
        net_keys = net_keys[net_range[0]:net_range[1]]
    print 'Computing performance for %d networks...' % len(net_keys)    
    
    for net_key in net_keys:
        print '----------- Computing SVM performance for %s ---------------' % net_key
        (test_loss_avg, test_loss_std) = train_readout_svm(net_file, net_key, readout_window=np.arange(0, 1))
        fnet[net_key].attrs['svm_perf'] = test_loss_avg
        fnet[net_key].attrs['svm_perf_std'] = test_loss_std
        print '%s: %0.3f +/- %0.3f' % (net_key, test_loss_avg, test_loss_std)
        
    fnet.close()

def compute_entropy_ratio(net_file):
    
    fnet = h5py.File(net_file, 'a')
    net_keys = fnet.keys()
    print 'Computing entropy ratio for %d networks...' % len(net_keys)    
    
    nbins_to_try = [8, 27, 64, 125, 216]
    
    for net_key in net_keys:
        eratios = []
        for nbins in nbins_to_try:
            eratio = np.nan
            try:            
                (eratio, uc_entropy, sc_entropy) = entropy_ratio(net_file, net_key, readout_window=np.arange(0, 1), nbins=nbins)
            except:
                print 'Problem with key %s' % net_key
            eratios.append([nbins, eratio])        
        fnet[net_key].attrs['entropy_ratio'] = np.array(eratios)
        
    fnet.close()

def get_perfs(net_files, entropy_ratio_bins=125):
    
    perf_types = ['nn', 'logit']
    perfs = []
    
    nbins_to_try = [8, 27, 64, 125, 216]
    nbins_index = nbins_to_try.index(entropy_ratio_bins)
    
    index2keys = []
    
    for net_file in net_files:
        print 'File: %s' % net_file
        f = h5py.File(net_file, 'r')
        net_keys = f.keys()
        nsamps = len(net_keys)
                
        for k,net_key in enumerate(net_keys):
            nn_perf = np.nan
            try:
                nn_perf = 1.0 / float(f[net_key].attrs['nn_perf'])                
            except:
                print '\tproblem getting NN perf from %s' % net_key
            logit_perf = np.nan
            try:
                logit_perf = 1.0 / float(f[net_key].attrs['logit_perf'])                
            except:
                print '\tproblem getting logit from %s' % net_key
            entropy_ratio = np.nan
            try:
                er = np.array(f[net_key].attrs['entropy_ratio'])
                entropy_ratio = er[nbins_index, 1]
            except:
                print '\tproblem getting entropy from %s' % net_key
            
            perfs.append([nn_perf, logit_perf, entropy_ratio])
            
            (rootdir, fname) = os.path.split(net_file)
            ikey = '%s[%s]' % (fname, net_key)
            index2keys.append(ikey)
            
        f.close()

    return (np.array(perfs), index2keys)
    