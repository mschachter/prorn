#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void step(__global float** state, __global float** params, __global float stepSize)
{
	const uint gpuId = get_global_id(0);
    printf("gpuId=%d\n", gpuId);
}