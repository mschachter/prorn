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
        self.total_len = None
        self.num_trials = None
        self.responses = None
        
    def to_hdf5(self, grp, compress=False):        
        grp.attrs['stim_start'] = self.stim_start
        grp.attrs['stim_end'] = self.stim_end
        grp.attrs['num_trials'] = self.num_trials
        grp['responses'] = self.responses

def run_sim(net, stims, burn_time=100, pre_stim_time = -1, post_stim_time=10, num_trials=1):
    
    net_sims = {}
    Ninternal = net.num_internal_nodes()
    for stim_key,stim in stims.iteritems():

        stimlen = len(stim)
        if pre_stim_time == -1:
            start_record = burn_time + stimlen            
        else:
            start_record = burn_time - pre_stim_time
        
        tlen_total = burn_time + stimlen + post_stim_time
        tlen_rec = tlen_total - start_record
        
        sim = Simulation()
        sim.net = net
        sim.stim_start = pre_stim_time
        sim.stim_end = sim.stim_start + stimlen
        sim.total_len = tlen_rec
        sim.num_trials = num_trials
        sim.responses = np.zeros([num_trials, tlen_rec, Ninternal])

        for n in range(num_trials):
            net_copy = copy.deepcopy(net)
            ais = ArrayInputStream(stim.reshape([len(stim), 1]))
            net_copy.set_input(ais)     
            net_copy.compile()
                    
            net_copy.set_stim_start_time(burn_time)
            
            for t in range(tlen_total):
                net_copy.step()
                trec = t - start_record
                if trec >= 0:
                    sim.responses[n, trec, :] = net_copy.x

        net_sims[stim_key] = sim

    return net_sims 
    