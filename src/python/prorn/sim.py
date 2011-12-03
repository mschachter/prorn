import copy

import h5py

from prorn.input import ArrayInputStream
from prorn.stim import read_stims_from_file

def run_sim(rnet, stim_file, output_file,
            burn_time=500, post_stim_time=500,
            start_record=450):

    fout = h5py.File(output_file, 'w') 
    
    stims = read_stims_from_file(stim_file)
    Ninternal = rnet.num_internal_nodes()    
    for stim_key,samples in stims.iteritems():
    
        stimlen = samples.shape[1]
        nsamps = samples.shape[0]
        tlen_total = burn_time + post_stim_time + stimlen
        tlen_rec = tlen_total - start_record        
        rnet_state = np.zeros([nsamps, tlen_rec, Ninternal])
        for k in range(nsamps):
            
            stim = samples[k, :]  
            ais = ArrayInputStream(stim.reshape([len(stim), 1]))
            
            rnet_copy = copy.deepcopy(rnet)
            rnet_copy.set_input(ais)     
            rnet_copy.compile()
                    
            rnet_copy.set_stim_start(burn_time)
            nrec = (burn_time - start_record) + stim_time
            
            for t in range(tlen_total):
                rnet_copy.step()
                trec = t - start_record
                if trec >= 0:
                    rnet_state[k, trec, :] = rnet_copy.x
        
        
            
    