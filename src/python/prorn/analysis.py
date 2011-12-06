import os
import csv
import operator

import numpy as np

import h5py

from prorn.readout import train_readout_nn, train_readout_logit, train_readout_svm
from prorn.info import entropy_ratio, mutual_information
from prorn.examples import run_many_nets

class PerformanceData:
    def __init__(self):
        self.file_name = None
        self.net_key = None
        self.nn_perf = None
        self.logit_perf = None
        self.entropy_ratio = None
        
        self.eigen_values = None        
        self.input_weight = None
    

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

def top100_weight_pca(pdata, filter=True, rootdir='/home/cheese63/git/prorn/data'):
    
    if filter:
        filter_perfs(pdata)
    
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
            
            nbins_to_try = [250, 300, 350, 400]
            
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
    


def get_perfs(net_files, entropy_ratio_bins=125):
    
    perf_types = ['nn', 'logit']
    perfs = []
    
    nbins_to_try = [8, 27, 64, 125, 216]
    nbins_index = nbins_to_try.index(entropy_ratio_bins)
    
    index2keys = []
    
    for net_file in net_files:
        if os.path.exists(net_file):
            print 'File: %s' % net_file
            f = h5py.File(net_file, 'r')
            net_keys = f.keys()
            nsamps = len(net_keys)
                    
            for k,net_key in enumerate(net_keys):
                nn_perf = np.nan
                try:
                    nn_perf = float(f[net_key].attrs['nn_perf'])                
                except:
                    print '\tproblem getting NN perf from %s' % net_key
                logit_perf = np.nan
                try:
                    logit_perf = float(f[net_key].attrs['logit_perf'])                
                except:
                    print '\tproblem getting logit from %s' % net_key
                entropy_ratio = np.nan
                try:
                    er = np.array(f[net_key].attrs['entropy_ratio'])
                    entropy_ratio = er[nbins_index, 1]
                except:
                    print '\tproblem getting entropy from %s' % net_key
                try:
                    W = np.array(f[net_key]['W'])
                    (evals, evecs) = np.linalg.eig(W)                    
                except:
                    print '\tProblem getting weights/eigenvalues from %s' % net_key
                
                perfs.append([nn_perf, logit_perf, entropy_ratio, evals[0], evals[1], evals[2]])
                
                (rootdir, fname) = os.path.split(net_file)
                ikey = (fname, net_key)
                index2keys.append(ikey)
                
            f.close()

    return (np.array(perfs), index2keys)

def write_perfs(net_files, perf_file, entropy_ratio_bins=125):
    
    f = open(perf_file, 'w')
    for net_file in net_files:
        if os.path.exists(net_file):
            (perfs, index2keys) = get_perfs([net_file], entropy_ratio_bins=entropy_ratio_bins)
            for k,(fname,net_key) in enumerate(index2keys):
                nn_perf = perfs[k, 0]
                logit_perf = perfs[k, 1]
                er = perfs[k, 2]
                ev1 = perfs[k, 3]
                ev2 = perfs[k, 4]
                ev3 = perfs[k, 5]
                f.write('%s,%s,%0.3f,%0.6f,%0.6f,%s,%s,%s\n' %
                        (fname, net_key, nn_perf, logit_perf, er,
                         str(ev1).strip(')('),str(ev2).strip(')('),str(ev3).strip(')(')))    
    f.close()   


def read_perfs(perf_file):
    
    pdata = []
    
    f = open(perf_file, 'r')
    cr = csv.reader(f)
    for row in cr:
        fname = str(row[0])
        net_key = str(row[1])
        nn_perf = float(row[2])
        logit_perf = float(row[3])
        er = float(row[4])
        ev1 = complex(row[5])
        ev2 = complex(row[6])
        ev3 = complex(row[7])
        
        evlist = [(ev1, np.abs(ev1)),
                  (ev2, np.abs(ev2)),
                  (ev3, np.abs(ev3))]
        evlist.sort(key=operator.itemgetter(1), reverse=True)
        
        ev1 = evlist[0][0]
        ev2 = evlist[1][0]
        ev3 = evlist[2][0]
        
        use = not (np.isnan(nn_perf) or np.isnan(logit_perf) or np.isnan(er)) 
        if use:
            pd = PerformanceData()
            pd.file_name = fname
            pd.net_key = net_key
            pd.nn_perf = nn_perf
            pd.logit_perf = logit_perf
            pd.entropy_ratio = er
            pd.eigen_values = (ev1, ev2, ev3)
            pdata.append(pd)
                
    f.close()
    
    return pdata


def filter_perfs(pdata, nn_perf_cutoff=70, logit_cutoff=0.17):
    
    pnew = []
    for p in pdata:
        if p.nn_perf < nn_perf_cutoff and p.logit_perf < logit_cutoff:
            pnew.append(p)
            
    return pnew

def create_many_networks(num_sets=5, nets_per_set=200, offset=3):
    
    for k in range(num_sets):
        net_file = '/home/cheese63/git/prorn/data/nets_0.99.%d.h5' % (k+1+offset)
        print '----------------------'
        print 'Creating %d nets in %s' % (nets_per_set, net_file)
        try:         
            run_many_nets(net_file, num_nets=nets_per_set, rescale_frac=0.99, index_offset=0)
        except:
            print '****  problem running nets on %s' % net_file
            continue
        try:        
            compute_readout_perf_nn(net_file)
        except:
            print '***** Problem computing nn perf on %s' % net_file
        try:
            compute_readout_perf_logit(net_file)
        except:
            print '***** Problem computing logit perf on %s' % net_file
        try:
            compute_entropy_ratio(net_file)
        except:
            print '***** Problem computing entropy on %s' % net_file
