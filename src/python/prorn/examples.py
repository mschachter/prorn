import numpy as np

import matplotlib.pyplot as plt

from prorn.network import *
from prorn.input import *
from prorn.stim import *

def create_inputless_net():
    
    rnet = EchoStateNetwork()
        
    rnet.create_node(1, initial_state=np.abs(np.random.randn()))
    rnet.create_node(2, initial_state=np.abs(np.random.randn()))
    rnet.create_node(3, initial_state=np.abs(np.random.randn()))
    
    rnet.connect_nodes(1, 2, 1.0)
    #rnet.connect_nodes(1, 3, 0.25)
    rnet.connect_nodes(2, 3, 1.1)
    rnet.connect_nodes(3, 1, 0.75)

    rnet.compile()
    
    return rnet

def create_fullyconnected_net():
    
    rnet = EchoStateNetwork()
    
    for n in [1, 2, 3]:        
        rnet.create_node(n, initial_state=np.abs(np.random.randn()))
        
    for n1 in [1, 2, 3]:
        for n2 in [1, 2, 3]:
            rnet.connect_nodes(n1, n2, np.random.randn())
    
    rnet.rescale_weights(0.95)
    
    return rnet

    
    

def create_3node_net(input_file='/home/cheese63/test.csv'):
    
    rnet = EchoStateNetwork()

    rnet.create_node(1)
    rnet.create_node(2)
    rnet.create_node(3)
    
    rnet.connect_nodes(1, 2, -0.5)
    rnet.connect_nodes(1, 3, -0.25)
    rnet.connect_nodes(2, 3, 0.9)
    rnet.connect_nodes(3, 1, -0.037)
    
    cis = CsvInputStream([1, 3], input_file)
    
    rnet.set_input(cis)    
    rnet.create_input(0)
    rnet.create_input(1)
    rnet.create_input(2)

    rnet.connect_input(0, 1, 0.25)
    rnet.connect_input(1, 2, 0.12)
    rnet.connect_input(2, 3, 0.72)
    

    rnet.compile()
    
    return rnet



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

    
    
    
    