import copy

import numpy as np

import h5py

from prorn.input import ArrayInputStream
from prorn.stim import read_stims_from_file

class Simulation():
    def __init__(self):
        self.net = None
        self.stim_start = None
        self.stim_end = None
        self.net_state = None
        

def run_sim(net, stims, burn_time=100, pre_stim_time = 10, post_stim_time=10):
    
    net_sims = {}
    Ninternal = net.num_internal_nodes()    
    for stim_key,samples in stims.iteritems():
        
        start_record = burn_time - pre_stim_time
        net_sims[sim_key] = []
        for stim in samples:
        
            stimlen = len(stim)
            tlen_total = burn_time + post_stim_time + stimlen
            tlen_rec = tlen_total - start_record        
            net_state = np.zeros([tlen_rec, Ninternal])
            
            ais = ArrayInputStream(stim.reshape([len(stim), 1]))
            
            net_copy = copy.deepcopy(net)
            net_copy.set_input(ais)     
            net_copy.compile()
                    
            net_copy.set_stim_start_time(burn_time)
            
            for t in range(tlen_total):
                net_copy.step()
                trec = t - start_record
                if trec >= 0:
                    net_state[trec, :] = net_copy.x
            
            sim = Simulation()
            sim.net = net_copy
            sim.stim_start = burn_time - start_record
            sim.stim_end = sim.stim_start + stimlen
            sim.net_state = net_state
            
            net_sims[sim_key].append(sim)
    
    return net_sims 
    