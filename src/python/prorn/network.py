import numpy as np
import networkx as nx

from prorn.input import CsvInputStream

class EchoStateNetwork:
    
    def __init__(self):
        self.net = nx.DiGraph()
        self.compiled = False
        self.input_stream = None
        self.node2index = {}
        self.index2node = []
        self.input_node2index = {}
        self.input_index2node = []
        
        self.t = 0
        self.W = None #weight matrix
        self.Win = None #input weight matrix
        self.x = None #state vector
        
    def set_stim_start_time(self, start_time):
        self.stim_start_time = start_time    
    
    def set_input(self, input_stream):
        """ Set an initialized input stream to read from. """
        self.input_stream = input_stream
        
    def create_node(self, id, initial_state=0.0):
        self.net.add_node(id, state=initial_state)
        
    def create_input(self, input_index):
        if input_index not in self.input_index2node:
            iname = 'input_%d' % input_index
            self.net.add_node(iname, index=input_index, is_input=True)
            self.input_node2index[iname] = len(self.input_node2index)
            self.input_index2node.append(iname)    
    
    def connect_nodes(self, id1, id2, weight=0.0):
        if id1 in self.net.nodes() and id2 in self.net.nodes():
            self.net.add_edge(id1, id2, weight=weight)
        else:
            print 'Edge not added, one of these nodes does not exist: %s or %s' % (str(id1), str(id2))
        
    def connect_input(self, input_index, n_id, weight=0.0):
        """ Connect an index in the input to a node """
        iname = self.input_index2node[input_index]
        self.net.add_edge(iname, n_id, weight=weight)
    
    def rescale_weights(self, frac=0.75):
        """ Rescale all weights so that they range between (-frac, frac) """
        
        max_weight = 0.0
        for e1,e2 in self.net.edges():
            w = abs(self.net[e1][e2]['weight'])
            if w > max_weight:
                max_weight = w
        
        for e1,e2 in self.net.edges():
            w = self.net[e1][e2]['weight']
            self.net[e1][e2]['weight'] = (w / max_weight)*frac
                
    def num_internal_nodes(self):
        return len(self.net.nodes()) - self.num_input_nodes()
    
    def num_input_nodes(self):
        x = np.array([1 for n in self.net.nodes() if 'is_input' in self.net.node[n]])
        return int(x.sum())
    
    def to_hdf5(self, f, key):
        """ Write this object to an hdf5 file """
        
        grp = f.create_group(key)
        grp['W'] = self.W
        grp['Win'] = self.Win        
        grp.attrs['type'] = '%s.%s' % (self.__module__, self.__class__.__name__)
    
    
    def compile(self):
        """ Create state vector and weight matrix. """
        
        internal_nodes = []
        input_nodes = []
        for n in self.net.nodes():
            if 'is_input' in self.net.node[n]:
                input_nodes.append(n)
            else:
                internal_nodes.append(n)
                
        self.t = 0
        N = len(internal_nodes)
        Nin = 0
        if self.input_stream is not None:
            Nin = max(self.input_stream.shape)
        self.x = np.zeros([N, 1], dtype='float').squeeze()
        self.W = np.zeros([N, N], dtype='float')
                
        if Nin != len(input_nodes):
            print '# of input nodes does not match pre-specified #: %d != %d' % (Nin, len(input_nodes))        
        
        #initialize state
        for k,n in enumerate(internal_nodes):
            self.node2index[n] = k
            self.index2node.append(n)            
            self.x[k] = self.net.node[n]['state']
        
        #initialize weight matrix
        for n1 in internal_nodes:
            if not 'is_input' in self.net.node[n1]:
                for n2,edge_attrs in self.net[n1].iteritems():                
                    i1 = self.node2index[n1]
                    i2 = self.node2index[n2]
                    self.W[i1, i2] = edge_attrs['weight']
        
        #initialize input
        if self.input_stream is not None and len(input_nodes) > 0:
            self.Win = np.zeros([N, Nin])
            for n1,n2 in self.net.edges():
                if 'is_input' in self.net.node[n1]:
                    input_index = self.input_node2index[n1]
                    node_index = self.node2index[n2]
                    self.Win[input_index, node_index] = self.net[n1][n2]['weight']                    
        
        self.compiled = True
        
    def step(self):
        """ Run a network through a single time step """
        
        if not self.compiled:
            self.compile()
        
        #get input
        i_input = 0.0
        if self.input_stream is not None and self.t >= self.stim_start_time:            
            input = self.input_stream.next()
            if input is not None:
                #compute weighted input for each node                
                i_input = np.dot(self.Win, input).squeeze()                
                
        #compute weighted input for each node
        i_internal = np.dot(self.W, self.x)
                
        self.x = i_input + i_internal           
        self.t += 1
        
    
    
        
        
        
        
        
    
    
    