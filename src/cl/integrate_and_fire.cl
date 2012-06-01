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

	const float R = params[pindex]; /* resistance */
	const float C = params[pindex+1]; /* capacitance */
	const float vthresh = params[pindex+2];  /* membrane potential threshold */
	const float vreset = params[pindex+3];   /* post-spike membrane potential reset */

	const float v = state[sindex];  /* membrane potential */
	const float spike_state = state[sindex+1]; /* spike=1.0 no spike=0.0 */

    const uint windex = weight_index[gpu_index];
    const uint nconn = num_conns[gpu_index];
/*
    printf("gpu_index=%d, sindex=%d, pindex=%d, windex=%d, num_conns=%d, step_size=%f\nR=%f,C=%f,vthresh=%f,vreset=%f, v=%f\n",
            gpu_index, sindex, pindex, windex, nconn, step_size, R, C, vthresh, vreset, state[sindex]);
*/

    /* reset spike state if just spiked */
    if (spike_state > 1e-3)
        next_state[sindex+1] = 0.0;

    if (v > vthresh) {

        next_state[sindex] = vreset;
        next_state[sindex+1] = 1.0;

    } else {
        float input = 0.0;
        uint pre_index;
        for (int k = 0; k < nconn; k++) {
            pre_index = conn_index[windex+k];
            /*
            printf("gpu_index=%d, weights[%d]=%0.6f, pre_index=%d, input state=%0.6f\n",
                   gpu_index, k, weights[windex+k], pre_index, state[pre_index]);
            */
            input += weights[windex+k]*state[pre_index];
        }

        next_state[sindex] = v + step_size*( (-v / (R*C)) + (input / C));
    }
}
