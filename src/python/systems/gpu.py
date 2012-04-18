
import numpy as np

import pyopencl as cl


kernel_fhn_cl = r"""
#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void step(__global const float *a, __global const float *b, __global float *c)
{
	const uint gpuId = get_global_id(0);
    printf("gpuId=%d\n", gpuId);
}
"""

kernel_test_cl = r"""
#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void step(__global const float *state_index, __global const float *param_index,
                   __global float *state, __global float *params,
                   __global float *weight_index,
                   __global float *num_conns, __global float *weights,
                   __global float *next_state,
                   __global float step_size)
{
	const uint gpu_index = get_global_id(0);
	const uint sindex = state_index[gpu_index];
	const uint pindex = param_index[gpu_index];

	const uint nstates = 2;
	const uint nparams = 2;

    const uint windex = weight_index[gpu_index];
    const uint nconn = num_conns[gpu_index];

    printf("gpu_index=%d, sindex=%d, pindex=%d, windex=%d, num_conns=%d\n",
            gpu_index, sindex, pindex, windex, num_conns);

    next_state[gpu_index] = ((float) gpu_index) + 2.5;
}

"""


class GpuUnit(object):
    def __init__(self):
        self.param_order = None
        self.params = None
        self.state = None
        self.kernel = None


class TestUnit(GpuUnit):
    def __init__(self):
        GpuUnit.__init__(self)

        self.param_order = ['a', 'b']
        self.params = {'a':1.0, 'b':5.0}
        self.state = [0.5, 0.75]
        self.kernel = kernel_test_cl


class GpuNetworkData(object):

    def __init__(self):

        self.unit2gpu = {}
        self.gpu2unit = {}

        self.unit_param_index = None
        self.params = None

        self.unit_state_index = None
        self.state = None
        self.next_state = None

        self.unit_weight_index = None
        self.weights = None
        self.num_connections = None


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
            for m,pname in enumerate(u.param_order):
                self.params[k+m] = u.params[pname]

            state_index += len(u.state)
            param_index += len(u.params)

        self.next_state = np.zeros(self.num_states)

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
        for uid,uconns in weight_map.iteritems():
            gpu_index = self.unit2gpu[uid]
            wlen = len(uconns)
            self.num_connections[gpu_index] = wlen
            self.unit_weight_index[gpu_index] = weight_index
            for m,pre_uid in enumerate(uconns):
                self.weights[weight_index+m] = weights[(pre_uid, uid)]

            weight_index += wlen

    def copy_to_gpu(self, cl_context):

        mf = cl.mem_flags
        self.unit_param_index_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.unit_param_index)
        self.params_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.params)

        self.unit_state_index_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.unit_state_index)
        self.state_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.state)
        self.next_state_buf = cl.Buffer(cl_context, mf.WRITE_ONLY, self.next_state.nbytes)

        self.unit_weight_index_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.unit_weight_index)
        self.weights_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.weights)
        self.num_connections_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.num_connections)

    def clear_gpu(self):
        pass

    def copy_from_gpu(self, cl_queue):
        cl.enqueue_copy(cl_queue, self.next_state, self.next_state_buf)


class GpuNetwork(object):

    def __init__(self, cl_context):
        self.units = []
        self.connections = {}
        self.network_data = None
        self.cl_context = cl_context

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
        self.network_data.init_weights(self.connections)

        print 'unit2gpu,',self.network_data.unit2gpu
        print 'gpu2unit,',self.network_data.gpu2unit

        print 'unit_param_index',self.network_data.unit_param_index
        print 'params',self.network_data.params

        print 'unit_state_index,',self.network_data.unit_state_index
        print 'state,',self.network_data.state
        print 'next_state,',self.network_data.next_state

        print 'unit_weight_index,',self.network_data.unit_weight_index
        print 'weights,',self.network_data.weights
        print 'num_connections,',self.network_data.num_connections

    def step(self, step_size):

        self.network_data.copy_to_gpu(self.cl_context)




def test_basic():

    gpunet = GpuNetwork()

    t1 = TestUnit()
    #t2 = TestUnit()

    gpunet.add_unit(t1)
    gpunet.compile()

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)



    """
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
    """
