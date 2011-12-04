import copy

import numpy as np

import h5py

from prorn.input import ArrayInputStream
from prorn.stim import read_stims_from_file

def run_sim(net, stims, burn_time=500, post_stim_time=500, start_record=450):
    
    net_stims = {}
    Ninternal = net.num_internal_nodes()    
    for stim_key,samples in stims.iteritems():
    
        stimlen = samples.shape[1]
        nsamps = samples.shape[0]
        tlen_total = burn_time + post_stim_time + stimlen
        tlen_rec = tlen_total - start_record        
        net_state = np.zeros([nsamps, tlen_rec, Ninternal])
        for k in range(nsamps):
            
            stim = samples[k, :]  
            ais = ArrayInputStream(stim.reshape([len(stim), 1]))
            
            net_copy = copy.deepcopy(net)
            net_copy.set_input(ais)     
            net_copy.compile()
                    
            net_copy.set_stim_start_time(burn_time)
                        
            for t in range(tlen_total):
                net_copy.step()
                trec = t - start_record
                if trec >= 0:
                    net_state[k, trec, :] = net_copy.x
        
        net_stims[stim_key] = net_state
    
    return net_stims 
    