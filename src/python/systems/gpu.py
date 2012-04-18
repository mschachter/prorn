
import numpy as np

import pyopencl as cl


fhn_cl = r"""
#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void step(__global const float *a, __global const float *b, __global float *c)
{
	const uint gpuId = get_global_id(0);
    printf("gpuId=%d\n", gpuId);
}
"""


class GpuUnit(object):
    def __init__(self):
        self.params = None
        self.state = None
        self.kernel = None


class GpuNetworkData(object):

    def __init__(self):

        self.unit2gpu = {}
        self.gpu2unit = {}

        self.unit_indices = None
        self.params = None
        self.state = None
        self.weights = None
        self.weight_index = None

    def init_units(self, units):
        num_params = 0
        num_states = 0
        for u in units:
            num_params += len(u.params)
            num_states += len(u.state)
        self.num_units = len(units)
        self.num_params = num_params
        self.num_states = num_states

        self.unit_state_index = np.zeros(self.num_units, dtype='int32')
        self.unit_param_index = np.zeros(self.num_units, dtype='int32')
        self.state = np.zeros(self.num_states, dtype='float32')
        self.params = np.zeros(self.num_params, dtype='float32')

        param_index = 0
        state_index = 0
        for k,u in enumerate(units):
            self.unit2gpu[u.id] = k
            self.gpu2unit[k] = u.id
            self.unit_state_index[k] = state_index
            self.unit_param_index[k] = param_index

            for m,s in enumerate(u.state):
                self.state[k+m] = s
            for m,p in enumerate(u.params):
                self.params[k+m] = p

            state_index += len(units.state)
            param_index += len(units.params)

    def init_weights(self, weights):

        weight_map = {}
        for (uid1,uid2),weight in weights.iteritems():
            if uid2 not in weight_map:
                weight_map[uid2] = []
            weight_map[uid2].append(uid1)

        total_num_conns = len(weights)
        self.num_connections = np.zeros(self.num_units, dtype='int32')
        self.unit_weight_index = np.zeros(self.num_units, dtype='int32')
        self.weights = np.zeros(total_num_conns, dtype='float32')
        weight_index = 0
        for k,uid in enumerate(weight_map.keys()):
            wlen = len(weight_map[uid])
            self.num_connections[k] = wlen
            self.unit_weight_index[k] = weight_index
            for m,w in enumerate(weight_map[uid]):
                self.weights[weight_index+m] = w

            weight_index += wlen






class GpuNetwork(object):

    def __init__(self):
        self.units = []
        self.connections = {}
        self.network_data = None

    def add_unit(self, u):
        uid = len(self.units)
        u.id = uid
        self.units.append(u)

    def connect(self, uid1, uid2, weight):
        ckey = (uid1, uid2)
        self.connections[ckey] = weight

    def compile(self):
        self.network_data = GpuNetworkData()
        self.network_data.init_units(self.units)









def test_fhn_neuron():
    a = np.random.rand(5).astype(np.float32)
    b = np.random.rand(5).astype(np.float32)
    
    ctx = cl.create_some_context()

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

    prg = cl.Program(ctx, fhn_cl).build()

    prg.step(queue, a.shape, None, a_buf, b_buf, dest_buf)

    a_plus_b = np.empty_like(a)
    cl.enqueue_copy(queue, a_plus_b, dest_buf)

    print np.linalg.norm(a_plus_b - (a+b))
