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

	const float v = state[sindex+1];  /* membrane potential */

    const uint windex = weight_index[gpu_index];
    const uint nconn = num_conns[gpu_index];
/*
    printf("gpu_index=%d, sindex=%d, pindex=%d, windex=%d, num_conns=%d, step_size=%f\nR=%f,C=%f,vthresh=%f,vreset=%f, v=%f\n",
            gpu_index, sindex, pindex, windex, nconn, step_size, R, C, vthresh, vreset, state[sindex]);
*/

    if (v > vthresh) {
        /* spike has occurred, reset membrane potential and set spike state to 1 */
        next_state[sindex+1] = vreset;
        next_state[sindex] = 1.0f;
    } else {
        /* integrate synaptic input */

        float input = 0.0f;
        float pre_state;
        int pre_index, pre_sindex;
        for (int k = 0; k < nconn; k++) {
            pre_index = conn_index[windex+k];
            pre_sindex = state_index[pre_index];
            pre_state = state[pre_sindex];  /* first state is always the observable one */
            /*
            printf("gpu_index=%d, weights[%d]=%0.6f, pre_index=%d, pre_sindex=%d, input state=%0.6f\n",
                   gpu_index, k, weights[windex+k], pre_index, pre_sindex, pre_state);
            */
            input += weights[windex+k]*pre_state;
        }
        /* update membrane potential and spike state */
        next_state[sindex+1] = v + step_size*( (-v / (R*C)) + (input / C));
        next_state[sindex] = 0.0f;
    }
}
