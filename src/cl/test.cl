#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void unit_step(__global const int *state_index, __global const int *param_index,
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
