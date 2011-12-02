import numpy as np

from prorn.network import *
from prorn.input import *

def create_inputless_net():
    
    rnet = ReservoirNetwork()
        
    rnet.create_node(1, initial_state=np.abs(np.random.randn()))
    rnet.create_node(2, initial_state=np.abs(np.random.randn()))
    rnet.create_node(3, initial_state=np.abs(np.random.randn()))
    
    rnet.connect_nodes(1, 2, 1.0)
    #rnet.connect_nodes(1, 3, 0.25)
    rnet.connect_nodes(2, 3, 1.1)
    rnet.connect_nodes(3, 1, 0.75)

    rnet.compile()
    
    return rnet
    

def create_3node_net(input_file='/home/cheese63/test.csv'):
    
    rnet = ReservoirNetwork()

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
