import numpy as np
import networkx as nx

from prorn.input import CsvInputStream

class EchoStateNetwork:
    
    def __init__(self):
        self.reinit()        
                
    def reinit(self):
        self.graph = nx.DiGraph()
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
        self.noise_std = 0.0
        
        
    def set_stim_start_time(self, start_time):
        self.stim_start_time = start_time    
    
    def set_input(self, input_stream):
        """ Set an initialized input stream to read from. """
        self.input_stream = input_stream
        
    def create_node(self, id, initial_state=0.0):
        self.graph.add_node(id, state=initial_state)
        
    def create_input(self, input_index):
        if input_index not in self.input_index2node:
            iname = 'input_%d' % input_index
            self.graph.add_node(iname, index=input_index, is_input=True)
            self.input_node2index[iname] = len(self.input_node2index)
            self.input_index2node.append(iname)
    
    def connect_nodes(self, id1, id2, weight=0.0):
        if id1 in self.graph.nodes() and id2 in self.graph.nodes():
            self.graph.add_edge(id1, id2, weight=weight)
        else:
            print 'Edge not added, one of these nodes does not exist: %s or %s' % (str(id1), str(id2))
        
    def connect_input(self, input_index, n_id, weight=0.0):
        """ Connect an index in the input to a node """
        iname = self.input_index2node[input_index]
        if iname in self.graph and n_id in self.graph[iname]:
            del self.graph[iname][n_id]
        self.graph.add_edge(iname, n_id, weight=weight)
    
    def rescale_weights(self, frac=0.75):
        """ Rescale all weights so that they range between (-frac, frac) """
        
        max_weight = 0.0
        for e1,e2 in self.graph.edges():
            w = abs(self.graph[e1][e2]['weight'])
            if w > max_weight:
                max_weight = w
        
        for e1,e2 in self.graph.edges():
            w = self.graph[e1][e2]['weight']
            self.graph[e1][e2]['weight'] = (w / max_weight)*frac
                
    def num_internal_nodes(self):
        return len(self.graph.nodes()) - self.num_input_nodes()
    
    def num_input_nodes(self):
        x = np.array([1 for n in self.graph.nodes() if 'is_input' in self.graph.node[n]])
        return int(x.sum())
    
    def to_hdf5(self, grp):
        """ Write this object to an hdf5 group """
        
        initial_state = []
        for n in self.graph.nodes():
            if not 'is_input' in self.graph.node[n]:
                initial_state.append(self.graph.node[n]['state'])
        
        N = self.num_internal_nodes()
        Nin = self.num_input_nodes()
        
        grp['W'] = self.W
        grp['Win'] = self.Win
        grp['initial_state'] = np.array(initial_state)
        grp.attrs['type'] = '%s.%s' % (self.__module__, self.__class__.__name__)
    
    def from_hdf5(self, grp):
        
        self.reinit()
        
        W = np.array(grp['W'])
        Win = np.array(grp['Win'])
        initial_state = np.array(grp['initial_state'])
        
        N = W.shape[0]
        Nin = Win.shape[1]
         
        #create internal nodes        
        for n in range(N):
            self.create_node(n, initial_state=initial_state[n])
            
        #create internal connections
        for n1 in range(N):
            for n2 in range(N):
                w = W[n1, n2]
                if w != 0.0:
                   self.connect_nodes(n1, n2, weight=w) 
        #create and connect inputs
        for nin in range(Nin):
            self.create_input(nin)
            for n in range(N):
                w = Win[n, nin]
                if w != 0.0:
                    self.connect_input(nin, n, weight=w)
        
    def compile(self):
        """ Create state vector and weight matrix. """
        
        internal_nodes = []
        input_nodes = []
        for n in self.graph.nodes():
            if 'is_input' in self.graph.node[n]:
                input_nodes.append(n)
            else:
                internal_nodes.append(n)
                
        self.t = 0
        N = len(internal_nodes)
        Nin = len(input_nodes)
        self.x = np.zeros([N, 1], dtype='float').squeeze()
        self.W = np.zeros([N, N], dtype='float')
                
        Nis = 0
        if self.input_stream is not None:
            Nis = max(self.input_stream.shape)
        if self.input_stream is not None and Nis != len(input_nodes):
            print '# of input nodes from input stream does not match pre-specified #: %d != %d' % (Nis, len(input_nodes))        
        
        #initialize state
        for k,n in enumerate(internal_nodes):
            self.node2index[n] = k
            self.index2node.append(n)            
            self.x[k] = self.graph.node[n]['state']
        
        #initialize weight matrix
        for n1 in internal_nodes:
            if not 'is_input' in self.graph.node[n1]:
                for n2,edge_attrs in self.graph[n1].iteritems():                
                    i1 = self.node2index[n1]
                    i2 = self.node2index[n2]
                    self.W[i1, i2] = edge_attrs['weight']
        
        #initialize input
        if len(input_nodes) > 0:
            self.Win = np.zeros([N, Nin])
            for n1,n2 in self.graph.edges():
                if 'is_input' in self.graph.node[n1]:
                    input_index = self.input_node2index[n1]
                    node_index = self.node2index[n2]
                    self.Win[input_index, node_index] = self.graph[n1][n2]['weight']                    
        
        self.compiled = True
        
    def step(self):
        """ Run a network through a single time step """
        
        if not self.compiled:
            self.compile()
        
        N = self.num_internal_nodes()
        
        #get input
        i_input = 0.0
        if self.input_stream is not None and self.t >= self.stim_start_time:            
            input = self.input_stream.next()
            if input is not None:
                #compute weighted input for each node                
                i_input = np.dot(self.Win, input).squeeze()                
                
        #compute weighted input for each node
        i_internal = np.dot(self.W, self.x)
        i_noise = 0.0
        if self.noise_std > 0.0:
            i_noise = np.random.randn(N)*self.noise_std
        self.x = i_input + i_internal + i_noise
        self.t += 1
        
    