__kernel void zero_mem(__global float* vals)
{
	const uint i = get_global_id(0);
	vals[i] += 1.2;
}
