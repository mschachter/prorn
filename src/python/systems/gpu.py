import copy
import os
import numpy as np

import pyopencl as cl

import matplotlib.pyplot as plt
import time

from prorn.config import get_root_dir


def read_cl(cl_name):
    cl_dir = os.path.join(get_root_dir(), 'src', 'cl')
    fname = os.path.join(cl_dir, cl_name)
    f = open(fname, 'r')
    cl_str = f.read()
    f.close()
    return cl_str

class GpuUnit(object):
    def __init__(self):
        self.param_order = None
        self.params = None
        self.state = None

class TestUnit(GpuUnit):
    CL_FILE = 'test.cl'

    def __init__(self):
        GpuUnit.__init__(self)

        self.param_order = ['a', 'b']
        self.params = {'a':1.0, 'b':5.0}
        self.state = [0.5, 0.75]


class IFUnit(GpuUnit):

    CL_FILE = 'integrate_and_fire.cl'

    def __init__(self):
        GpuUnit.__init__(self)

        self.param_order = ['R', 'C', 'vthresh', 'vreset']
        self.params = {'R':1.0, 'C':1e-2, 'vthresh':1.0, 'vreset':0.0}
        self.state = [0.0, 0.0] #spike/nospike, membrane potential

KERNELS = {'TestUnit': read_cl(TestUnit.CL_FILE),
           'IFUnit': read_cl(IFUnit.CL_FILE)}


class GpuInputStream(object):

    def __init__(self):
        self.ndim = None

    def pull(self, t):
        return None

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
        self.stream_uids = {}
        self.uids_stream = {}

        self.total_state_size = None


    def init_units(self, units, streams, stream_uids):

        self.streams = streams

        num_params = 0
        num_states = 0
        for u in units:
            num_params += len(u.params)
            num_states += len(u.state)

        self.stream_uids = stream_uids
        self.num_units = len(units)
        self.num_params = num_params
        self.num_states = num_states

        self.unit_state_index = np.zeros(self.num_units+len(self.stream_uids), dtype='int32')
        self.unit_param_index = np.zeros(self.num_units, dtype='int32')
        self.state = np.zeros(self.num_states+len(self.stream_uids), dtype='float32')
        self.params = np.zeros(self.num_params, dtype='float32')

        #assign states and parameters to the unit state and parameter arrays
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

        #map the stream inputs to gpu indices
        for k,((stream_id,stream_index),suid) in enumerate(self.stream_uids.iteritems()):
            self.uids_stream[suid] = (stream_id,stream_index)
            gpu_index = self.num_units + k
            self.gpu2unit[gpu_index] = suid
            self.unit2gpu[suid] = gpu_index
            self.unit_state_index[gpu_index] = state_index
            state_index += 1

        self.total_state_size = state_index

        #set up an array to store the next states
        self.next_state = np.zeros(self.num_states)

    def init_weights(self, weights):

        weight_map = {}
        for (uid1,uid2),weight in weights.iteritems():
            if uid2 not in weight_map:
                weight_map[uid2] = []
            weight_map[uid2].append(uid1)

        total_num_conns = len(weights)
        self.num_connections = np.zeros(self.num_units, dtype='int32') #holds the number of input connections for each unit
        self.unit_weight_index = np.zeros(self.num_units, dtype='int32') #the index into the weight array for each unit
        self.weights = np.zeros(total_num_conns, dtype='float32') #the actual weights
        self.conn_index = np.zeros(total_num_conns, dtype='int32') #holds the indices of input connections for each unit
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

    def update_state(self, cl_context, cl_queue, time):
        del self.next_state
        self.next_state = np.empty(self.num_states, dtype='float32')

        mf = cl.mem_flags
        cl.enqueue_copy(cl_queue, self.next_state, self.next_state_buf)

        self.state_buf.release()
        del self.state

        self.state = np.zeros([self.total_state_size], dtype='float32')
        self.state[:self.num_states] = self.next_state
        for s in self.streams:
            sval = s.pull(time)
            for sindex in range(s.ndim):
                skey = (s.id, sindex)
                suid = self.stream_uids[skey]
                gpu_index = self.unit2gpu[suid]
                state_index = self.unit_state_index[gpu_index]
                self.state[state_index] = sval[sindex]
        self.state_buf = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.state)

    def clear(self):

        self.unit_param_index_buf.release()
        del self.unit_param_index_buf
        self.params_buf.release()
        del self.params_buf

        self.unit_state_index_buf.release()
        del self.unit_state_index_buf
        self.state_buf.release()
        del self.state_buf
        self.next_state_buf.release()
        del self.next_state_buf

        self.unit_weight_index_buf.release()
        del self.unit_weight_index_buf
        self.weights_buf.release()
        del self.weights_buf
        self.conn_index_buf.release()
        del self.conn_index_buf
        self.num_connections_buf.release()
        del self.num_connections_buf


class GpuNetwork(object):

    def __init__(self, cl_context):
        self.units = []
        self.connections = {}
        self.stream_connections = {}
        self.network_data = None
        self.cl_context = cl_context
        self.step_size_buf = None
        self.streams = []
        self.stream_uids = {}
        self.time = 0.0

    def add_stream(self, istream):
        self.streams.append(istream)
        stream_id = len(self.streams)-1
        istream.id = stream_id
        for k in range(istream.ndim):
            skey = (istream.id, k)
            uid = -(len(self.stream_uids)+1)
            self.stream_uids[skey] = uid

    def add_unit(self, u):
        uid = len(self.units)
        u.id = uid
        self.units.append(u)

    def connect(self, u1, u2, weight):
        ckey = (u1.id, u2.id)
        self.connections[ckey] = weight

    def connect_stream(self, stream, stream_index, unit, weight):
        """ Connect an element of an input stream to a unit """
        skey = (stream.id, stream_index)
        suid = self.stream_uids[skey]
        ckey = (suid, unit.id)
        self.connections[ckey] = weight

    def compile(self):
        self.network_data = GpuNetworkData()
        self.network_data.init_units(self.units, self.streams, self.stream_uids)
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
        print 'conn_index,',self.network_data.conn_index
        print 'num_connections,',self.network_data.num_connections



    def step(self, step_size):

        global_size =  (len(self.units), )
        self.program.unit_step(self.queue, global_size, None,
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

        self.time += step_size
        self.network_data.update_state(self.cl_context, self.queue, self.time)
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



class ConstantInputStream(GpuInputStream):

    def __init__(self, amp, start, stop):
        GpuInputStream.__init__(self)
        self.ndim = 1
        self.start = start
        self.stop = stop
        self.amp = amp

    def pull(self, t):
        if t >= self.start and t < self.stop:
            return np.array([self.amp])
        else:
            return np.array([0.0])


def test_if(nunits=10, sim_dur=0.500):

    ctx = cl.create_some_context()
    gpunet = GpuNetwork(ctx)

    instream = ConstantInputStream(1.00, 0.020, 0.150)
    gpunet.add_stream(instream)

    for k in range(nunits):
        u = IFUnit()
        gpunet.add_unit(u)

    u0 = gpunet.units[0]
    gpunet.connect_stream(instream, 0, u0, 1.75)

    for k,u in enumerate(gpunet.units[1:]):
        uprev = gpunet.units[k]
        #w = 1.00 / float(k+2)
        #gpunet.connect_stream(instream, 0, u, w)
        gpunet.connect(uprev, u, 1.0)

    gpunet.compile()

    stime = time.time()
    step_size = 0.00025
    nsteps = int(sim_dur / step_size)
    all_states = []
    for k in range(nsteps):
        state = gpunet.step(step_size)
        #print 'k=%d, state=%s' % (k, str(state))
        all_states.append(state)
    etime = time.time() - stime
    print '%0.1fs to stimulate %0.3fs' % (etime, sim_dur)

    gpunet.clear()

    all_states = np.array(all_states)
    plt.figure()
    t = np.arange(0.0, sim_dur, step_size)
    for k in range(nunits):
        uindex = k*2
        st = all_states[:, uindex]
        v = all_states[:, uindex+1]
        #print 'k=%d, # of spikes=%d' % (k, st.sum())
        v[st > 0.0] = 3.0

        plt.plot(t, v)
