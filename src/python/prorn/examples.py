import h5py
import copy
import numpy as np

import matplotlib.pyplot as plt

from prorn.network import EchoStateNetwork
from prorn.input import NullInputStream
from prorn.sim import Simulation, run_sim
from prorn.readout import train_readout_mnlogit

from prorn.morse import MorseStimSet

RANDOM_SEED = 32588987237

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

def create_fullyconnected_net(num_nodes, rescale_frac=0.999, noise_std=0.0, random_initial_state=False,
                              symmetric=False, feedforward=False, connection_rate=1.0):
    
    net = EchoStateNetwork()
    net.noise_std = noise_std
    
    for k in range(num_nodes):
        n = k
        istate = 0.0       
        if random_initial_state:
            istate = np.abs(np.random.randn()) 
        net.create_node(n, initial_state=istate)
    
    W = np.random.randn(num_nodes, num_nodes)
    if symmetric:
        for k1 in range(num_nodes):
            for k2 in range(k1):
                W[k2, k1] = W[k1, k2]
    
    if feedforward:
        for k1 in range(num_nodes):
            for k2 in range(k1+1):
                W[k1, k2] = 0.0
    
    for k1 in range(num_nodes):
        for k2 in range(num_nodes):
            rn = np.random.rand()
            if rn < connection_rate:
                net.connect_nodes(k1, k2, W[k1, k2])
    
    net.rescale_weights(rescale_frac)
    
    return net
   


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
    
def run_morse_nets(stim_file, net_file, input_gain=1.0, noise_std=0.0, num_trials=1, exp_desc='default', net_keys=None, fixed_seed=None):

    stimset = MorseStimSet()
    stimset.from_hdf5(stim_file)
    f = h5py.File(net_file, 'a')
    
    burn_time = 100
    post_stim_time = 30
    pre_stim_time = 30 
    
    if net_keys is None:
        net_keys = f.keys()
    
    for net_key in net_keys:
        if fixed_seed is not None:
            #use a fixed random seed so each network sees the same set of random stimuli
            print 'Using fixed random seed..'
            np.random.seed(fixed_seed)
            
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

def run_morse_nets_online(stim_file, num_trials=20, stack_size=5,
                          num_nodes=3, input_gain=1.0, noise_std=0.0, fixed_seed=None,
                          stim_class='standard', num_nets=10, output_file=None,
                          symmetric=False, feedforward=False, connection_rate=1.0,
                          output_nl_type=None, output_nl_args=None):
    
    stimset = MorseStimSet()
    stimset.from_hdf5(stim_file)
   
    burn_time = 100
    post_stim_time = 1
    pre_stim_time = -1
    
    if fixed_seed is not None:
        #use a fixed random seed so each network sees the same set of random stimuli
        print 'Using fixed random seed..'
        np.random.seed(fixed_seed)
    
    #get stims
    stim_md5s = stimset.class_to_md5[stim_class]
    all_stims = {}
    for md5 in stim_md5s:
        all_stims[md5] = stimset.all_stims[md5]
        
    net_stack = [(None, None, None)]*stack_size
    net_performances = np.zeros([stack_size])
        
    for k in range(num_nets):
            
        #read network weights from hdf5
        net = create_fullyconnected_net(num_nodes=num_nodes, symmetric=symmetric,
                                        feedforward=feedforward)        
        net.noise_std = noise_std
        if output_nl_type is not None:
            output_nl = output_nl_type(net, **output_nl_args)
            net.output_nl = output_nl
        
        #try out each input node, see which one works best
        best_performance = -np.Inf
        best_input_node = 0
        best_sims = None

        for input_node in range(num_nodes):
            if feedforward and input_node != 0:
                continue
            
            net_copy = copy.deepcopy(net)
            net_copy.create_input(0)
            net_copy.connect_input(0, input_node, input_gain)
            net_copy.compile()
            net_sims = run_sim(net_copy, all_stims,
                               burn_time=burn_time,
                               pre_stim_time=pre_stim_time, post_stim_time=post_stim_time,
                               num_trials=num_trials)
            
            all_samps = {}
            for md5,sim in net_sims.iteritems():
                all_samps[md5] = sim.responses[:, 0, :].squeeze()                
            
            percent_correct = train_readout_mnlogit(stimset, all_samps)
            if percent_correct > best_performance:
                best_performance = percent_correct
                best_input_node = input_node
                best_net = net_copy
                best_sims = net_sims
                
        print 'Net %d (input=%d) got %0.2f correct' % (k, best_input_node, best_performance)
        """
        for md5,sim in best_sims.iteritems():
            sp = (sim.responses[:, 0, :].squeeze()).sum(axis=0) / float(len(all_samps[md5]))
            spstr = ['%d' % f for f in sp] 
            print '%s,%s' % (''.join(spstr), stimset.md5_to_symbol[md5])
        """
        
        min_perf = net_performances.min()
        if min_perf < best_performance:
            min_perf_index = net_performances.argmin()
            net_stack[min_perf_index] = (net, best_input_node, best_net)
            net_performances[min_perf_index] = best_performance

    if output_file is not None:
        
        f = h5py.File(output_file, 'w')
        for k,(sim, input_node, net) in enumerate(net_stack):
            perf = net_performances[k]
            net_name = 'net_%d' % k
            
            net_grp = f.create_group(net_name)
            net.to_hdf5(net_grp)
            
            perf_grp = net_grp.create_group('performance')
            perf_grp[stim_class] = perf
        
        f.close()
    

    return (net_stack, net_performances)
    

def run_morse_experiments(stim_file, net_file, num_trials=15):
    #run_morse_nets(stim_file, net_file, input_gain=1.0, noise_std=0.0, num_trials=1, exp_desc='noise_std_0.0-input_gain_1.0')
    #run_morse_nets(stim_file, net_file, input_gain=0.5, noise_std=0.0, num_trials=1, exp_desc='noise_std_0.0-input_gain_0.5')
    run_morse_nets(stim_file, net_file, input_gain=1.0, noise_std=0.10, num_trials=num_trials, exp_desc='noise_std_0.10-input_gain_1.0', fixed_seed=RANDOM_SEED)
    
    