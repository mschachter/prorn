import copy
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

__kernel void step(__global const int *state_index, __global const int *param_index,
                   __global const float *state, __global const float *params,
                   __global const int *weight_index, __global const int *conn_index,
                   __global const int *num_conns, __global const float *weights,
                   __global float *next_state,
                   const float step_size)
{
	const uint gpu_index = get_global_id(0);
	const uint sindex = state_index[gpu_index];
	const uint pindex = param_index[gpu_index];

	const uint nstates = 2;
	const uint nparams = 2;

    const uint windex = weight_index[gpu_index];
    const uint nconn = num_conns[gpu_index];

    /*
    printf("gpu_index=%d, sindex=%d, pindex=%d, windex=%d, num_conns=%d, step_size=%f\n",
            gpu_index, sindex, pindex, windex, nconn, step_size);
    */

    float input = 0.0;
    for (int k = 0; k < nconn; k++)
        input += weights[windex+k]*state[conn_index[windex+k]];

    next_state[sindex] = state[sindex] + 1.0;
    next_state[sindex+1] = input;
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


KERNELS = {'TestUnit': kernel_test_cl}


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
        self.conn_index = None
        self.num_connections = None

        self.unit_types = {}


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

            cname = u.__class__.__name__
            if cname not in self.unit_types:
                self.unit_types[cname] = []
            self.unit_types[cname].append(u.id)

            for m,s in enumerate(u.state):
                self.state[state_index+m] = s
            for m,pname in enumerate(u.param_order):
                self.params[param_index+m] = u.params[pname]

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
        self.conn_index = np.zeros(total_num_conns, dtype='float32')
        weight_index = 0
        for uid,uconns in weight_map.iteritems():
            gpu_index = self.unit2gpu[uid]
            wlen = len(uconns)
            self.num_connections[gpu_index] = wlen
            self.unit_weight_index[gpu_index] = weight_index
            for m,pre_uid in enumerate(uconns):
                pre_gpu_index = self.unit2gpu[pre_uid]
                self.weights[weight_index+m] = weights[(pre_uid, uid)]
                self.conn_index[weight_index+m] = pre_gpu_index

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
        self.conn_index_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.conn_index)
        self.num_connections_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.num_connections)

    def update_state(self, cl_context, cl_queue):
        del self.next_state
        self.next_state = np.empty_like(self.state)

        mf = cl.mem_flags
        cl.enqueue_copy(cl_queue, self.next_state, self.next_state_buf)

        self.state_buf.release()
        del self.state
        self.state = copy.copy(self.next_state)
        self.state_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.state)

    def clear(self):
        self.unit_param_index_buf.release()
        self.params_buf.release()

        self.unit_state_index_buf.release()
        self.state_buf.release()
        self.next_state_buf.release()

        self.unit_weight_index_buf.release()
        self.weights_buf.release()
        self.num_connections_buf.release()




class GpuNetwork(object):

    def __init__(self, cl_context):
        self.units = []
        self.connections = {}
        self.network_data = None
        self.cl_context = cl_context
        self.step_size_buf = None

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

        if len(self.network_data.unit_types) > 1:
            print 'ERROR: only homogeneous unit types are currently allowed!'
            return

        self.kernel_cl = KERNELS[self.network_data.unit_types.keys()[0]]
        self.program = cl.Program(self.cl_context, self.kernel_cl).build()
        self.kernel = self.program.all_kernels()[0]

        self.queue = cl.CommandQueue(self.cl_context)

        self.network_data.copy_to_gpu(self.cl_context)

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

        global_size =  (len(self.units), )
        self.program.step(self.queue, global_size, None,
                          self.network_data.unit_state_index_buf,
                          self.network_data.unit_param_index_buf,
                          self.network_data.state_buf,
                          self.network_data.params_buf,
                          self.network_data.unit_weight_index_buf,
                          self.network_data.conn_index_buf,
                          self.network_data.num_connections_buf,
                          self.network_data.weights_buf,
                          self.network_data.next_state_buf,
                          np.float32(step_size))

        self.network_data.update_state(self.cl_context, self.queue)
        return self.network_data.state


    def clear(self):
        self.network_data.clear()


def print_device_info():

    ctx = cl.create_some_context()
    devices = ctx.get_info(cl.context_info.DEVICES)
    device = devices[0]

    print 'Vendor: %s' % device.vendor
    print 'Name: %s' % device.name
    print 'Max Clock Freq: %0.0f' % device.max_clock_frequency
    gmem = float(device.global_mem_size) / 1024**2
    print 'Global Memory: %0.0f MB' % gmem
    print '# of Compute Units: %d' % device.max_compute_units


def test_basic():

    ctx = cl.create_some_context()

    gpunet = GpuNetwork(ctx)

    t1 = TestUnit()
    t1.state = [0.25, 0.5]
    t2 = TestUnit()
    t2.state = [0.7, 0.9]

    gpunet.add_unit(t1)
    gpunet.add_unit(t2)
    gpunet.connect(t1.id, t2.id, 0.5)

    gpunet.compile()

    nsteps = 5
    for k in range(nsteps):
        state = gpunet.step(0.0005)
        print 'k=%d, state=%s' % (k, str(state))

    gpunet.clear()

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
