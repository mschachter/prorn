import h5py
import numpy as np

import matplotlib.pyplot as plt

from prorn.network import EchoStateNetwork
from prorn.input import NullInputStream
from prorn.sim import Simulation, run_sim

from prorn.morse import MorseStimSet

def create_inputless_net():
    
    net = EchoStateNetwork()
        
    net.create_node(1, initial_state=np.abs(np.random.randn()))
    net.create_node(2, initial_state=np.abs(np.random.randn()))
    net.create_node(3, initial_state=np.abs(np.random.randn()))
    
    net.connect_nodes(1, 2, 1.0)
    #net.connect_nodes(1, 3, 0.25)
    net.connect_nodes(2, 3, 1.1)
    net.connect_nodes(3, 1, 0.75)

    net.compile()
    
    return net

def create_fullyconnected_net(num_nodes, rescale_frac=0.999, noise_std=0.0, random_initial_state=False):
    
    net = EchoStateNetwork()
    net.noise_std = noise_std
    
    for k in range(num_nodes):
        n = k
        istate = 0.0       
        if random_initial_state:
            istate = np.abs(np.random.randn()) 
        net.create_node(n, initial_state=istate)
        
    for k1 in range(num_nodes):
        for k2 in range(num_nodes):
            net.connect_nodes(k1, k2, np.random.randn())
    
    net.rescale_weights(rescale_frac)
    
    return net
   

def create_3node_net(input_file='/home/cheese63/test.csv'):
    
    net = EchoStateNetwork()

    net.create_node(1)
    net.create_node(2)
    net.create_node(3)
    
    net.connect_nodes(1, 2, -0.5)
    net.connect_nodes(1, 3, -0.25)
    net.connect_nodes(2, 3, 0.9)
    net.connect_nodes(3, 1, -0.037)
    
    cis = CsvInputStream([1, 3], input_file)
    
    net.set_input(cis)    
    net.create_input(0)
    net.create_input(1)
    net.create_input(2)

    net.connect_input(0, 1, 0.25)
    net.connect_input(1, 2, 0.12)
    net.connect_input(2, 3, 0.72)
    

    net.compile()
    
    return net



def show_stims():
    
    stim_len = 10
    t = np.arange(0, stim_len)
    
    bump_times = [5,3]
    bump_heights = [1,0.5]
    bump_widths = [0.1,0.05]
    nsamps = 5
    
    stims = gen_1d_stim_family(stim_len, bump_times, bump_heights, bump_widths, num_samps=nsamps)
    
    for k in range(nsamps):
        plt.plot(t, stims[k, :])
        
    plt.show()

    
def run_example_net():
    
    net = create_fullyconnected_net()
    stims = read_stims_from_file('/home/cheese63/git/prorn/data/stims.h5')    
    net_stims = run_sim(net, stims)
    
    return net_stims

            
def run_many_nets(output_file, num_nets=5, rescale_frac=0.75, index_offset=0):
    
    stims = read_stims_from_file('/home/cheese63/git/prorn/data/stims.h5')
    
    f = h5py.File(output_file, 'a')
    
    nis = NullInputStream([1, 1])
    
    burn_time = 100
    post_stim_time = 20
    pre_stim_record = 10 
    start_record = burn_time - pre_stim_record
    stim_start = burn_time - start_record #relative to recording time start
    
    for k in range(num_nets):
        
        net_num =k + index_offset
        print 'Running net %d...' % net_num
        net_key = 'net_%d' % net_num
        net = create_fullyconnected_net(rescale_frac=rescale_frac)
        net_stims = run_sim(net, stims,
                             burn_time=burn_time, post_stim_time=post_stim_time, start_record=start_record)
         
        net.set_input(nis)
        net.compile()
        net.to_hdf5(f, net_key)
         
        for stim_key,net_state in net_stims.iteritems():
            
            stimlen = stims[stim_key].shape[1]
            stim_end = stim_start + stimlen            
            f[net_key][stim_key] = net_state
            f[net_key][stim_key].attrs['stim_start'] = stim_start
            f[net_key][stim_key].attrs['stim_end'] = stim_end
    
    f.close()
    

def create_morse_networks(net_file, num_nets=1, index_offset=0, num_nodes=3, input_gain=1.0):
    
    f = h5py.File(net_file, 'a')
    
    for k in range(num_nets):
        
        net_num = k + index_offset
        print 'Creating net %d...' % net_num
        net_key = 'net_%d' % net_num
        
        net = create_fullyconnected_net(num_nodes=num_nodes)
        
        net.create_input(0)
        net.connect_input(0, 0, input_gain)        
        
        net.compile()
        
        net_grp = f.create_group(net_key)
        net.to_hdf5(net_grp)
    
    f.close()
    
def run_morse_nets(stim_file, net_file, input_gain=1.0, noise_std=0.0, num_trials=1, exp_desc='default', net_keys=None):

    stimset = MorseStimSet()
    stimset.from_hdf5(stim_file)
    f = h5py.File(net_file, 'a')
    
    burn_time = 100
    post_stim_time = 30
    pre_stim_time = 30 
    
    if net_keys is None:
        net_keys = f.keys()
    
    for net_key in net_keys:
        
        print 'Running network %s' % net_key
        net_grp = f[net_key]
        
        #read network weights from hdf5
        net = EchoStateNetwork()
        net.from_hdf5(net_grp)
        net.noise_std = noise_std
        
        #set the input gain
        net.connect_input(0, 0, input_gain)        
        
        #rebuild network and run
        net.compile()        
        net_sims = run_sim(net, stimset.all_stims,
                           burn_time=burn_time,
                           pre_stim_time=pre_stim_time, post_stim_time=post_stim_time,
                           num_trials=num_trials)

        #save experiment to network
        exp_grp = net_grp.create_group(exp_desc)
        exp_grp['noise_std'] = noise_std
        exp_grp['num_stims'] = len(net_sims)
        exp_grp['input_gain'] = input_gain
        
        resp_grp = exp_grp.create_group('responses')

        for stim_key,sim in net_sims.iteritems():
            stim_grp = resp_grp.create_group(stim_key)
            sim.to_hdf5(stim_grp)

    f.close()

def run_morse_experiments(stim_file, net_file):
    #run_morse_nets(stim_file, net_file, input_gain=1.0, noise_std=0.0, num_trials=1, exp_desc='noise_std_0.0-input_gain_1.0')
    #run_morse_nets(stim_file, net_file, input_gain=0.5, noise_std=0.0, num_trials=1, exp_desc='noise_std_0.0-input_gain_0.5')
    run_morse_nets(stim_file, net_file, input_gain=1.0, noise_std=0.10, num_trials=15, exp_desc='noise_std_0.10-input_gain_1.0')
    
    