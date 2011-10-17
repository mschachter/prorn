#ifndef _VIENNACL_VECTOR_SOURCE_HPP_
#define _VIENNACL_VECTOR_SOURCE_HPP_
//Automatically generated file from aux-directory, do not edit manually!
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
const char * const vector_align16_cpu_inplace_mul = 
"\n"
"__kernel void cpu_inplace_mult(\n"
"          __global float16 * vec,\n"
"          float factor, \n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size/16; i += get_global_size(0))\n"
"    vec[i] *= factor;\n"
"}\n"
"\n"
; //vector_align16_cpu_inplace_mul

const char * const vector_align16_cpu_mult = 
"__kernel void cpu_mult(\n"
"          __global const float16 * vec,\n"
"          float factor, \n"
"          __global float16 * result,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size/16; i += get_global_size(0))\n"
"    result[i] = vec[i] * factor;\n"
"}\n"
; //vector_align16_cpu_mult

const char * const vector_align16_sub = 
"__kernel void sub(\n"
"          __global const float16 * vec1,\n"
"          __global const float16 * vec2, \n"
"          __global float16 * result,\n"
"          unsigned int size)\n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size/16; i += get_global_size(0))\n"
"    result[i] = vec1[i] - vec2[i];\n"
"}\n"
; //vector_align16_sub

const char * const vector_align16_inplace_mult = 
"__kernel void inplace_mult(\n"
"          __global float16 * vec,\n"
"          __global const float * fac, \n"
"          unsigned int size) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size/16; i += get_global_size(0))\n"
"    vec[i] *= factor;\n"
"}\n"
; //vector_align16_inplace_mult

const char * const vector_align16_divide = 
"//Note: 'div' cannot be used because of complaints by the jit-compiler\n"
"__kernel void divide(\n"
"          __global const float16 * vec,\n"
"          __global const float * fac,  //note: CPU variant is mapped to prod_scalar\n"
"          __global float16 * result,\n"
"          unsigned int size)  \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size/16; i += get_global_size(0))\n"
"    result[i] = vec[i] / factor;\n"
"}\n"
; //vector_align16_divide

const char * const vector_align16_add = 
"__kernel void add(\n"
"          __global const float16 * vec1,\n"
"          __global const float16 * vec2, \n"
"          __global float16 * result,\n"
"          unsigned int size)\n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size/16; i += get_global_size(0))\n"
"    result[i] = vec1[i] + vec2[i];\n"
"}\n"
; //vector_align16_add

const char * const vector_align16_inplace_divide = 
"__kernel void inplace_divide(\n"
"          __global float16 * vec,\n"
"          __global const float * fac,  //note: CPU variant is mapped to prod_scalar\n"
"          unsigned int size) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size/16; i += get_global_size(0))\n"
"    vec[i] /= factor;\n"
"}\n"
; //vector_align16_inplace_divide

const char * const vector_align16_inplace_sub = 
"__kernel void inplace_sub(\n"
"          __global float16 * vec1,\n"
"          __global const float16 * vec2,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size/16; i += get_global_size(0))\n"
"    vec1[i] -= vec2[i];\n"
"}\n"
; //vector_align16_inplace_sub

const char * const vector_align16_inplace_add = 
"__kernel void inplace_add(\n"
"          __global float16 * vec1,\n"
"          __global const float16 * vec2,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size/16; i += get_global_size(0))\n"
"    vec1[i] += vec2[i];\n"
"}\n"
; //vector_align16_inplace_add

const char * const vector_align16_mult = 
"__kernel void mult(\n"
"          __global const float16 * vec,\n"
"          __global const float * fac, \n"
"          __global float16 * result,\n"
"          unsigned int size) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size/16; i += get_global_size(0))\n"
"    result[i] = vec[i] * factor;\n"
"}\n"
; //vector_align16_mult

const char * const vector_align1_plane_rotation = 
"////// plane rotation: (x,y) <- (\alpha x + \beta y, -\beta x + \alpha y)\n"
"__kernel void plane_rotation(\n"
"          __global float * vec1,\n"
"          __global float * vec2, \n"
"          float alpha,\n"
"          float beta,\n"
"          unsigned int size) \n"
"{ \n"
"  float tmp1 = 0;\n"
"  float tmp2 = 0;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"  {\n"
"    tmp1 = vec1[i];\n"
"    tmp2 = vec2[i];\n"
"    \n"
"    vec1[i] = alpha * tmp1 + beta * tmp2;\n"
"    vec2[i] = alpha * tmp2 - beta * tmp1;\n"
"  }\n"
"}\n"
; //vector_align1_plane_rotation

const char * const vector_align1_inplace_mul_add = 
"__kernel void inplace_mul_add(\n"
"          __global float * vec1,\n"
"          __global const float * vec2,\n"
"          __global const float * fac,\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    vec1[i] += vec2[i] * factor;\n"
"}\n"
; //vector_align1_inplace_mul_add

const char * const vector_align1_index_norm_inf = 
"//index_norm_inf:\n"
"unsigned int float_vector1_index_norm_inf_impl(\n"
"          __global const float * vec,\n"
"          unsigned int size,\n"
"          __local float * float_buffer,\n"
"          __local unsigned int * index_buffer)\n"
"{\n"
"  //step 1: fill buffer:\n"
"  float cur_max = 0.0f;\n"
"  float tmp;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"  {\n"
"    tmp = fabs(vec[i]);\n"
"    if (cur_max < tmp)\n"
"    {\n"
"      float_buffer[get_global_id(0)] = tmp;\n"
"      index_buffer[get_global_id(0)] = i;\n"
"      cur_max = tmp;\n"
"    }\n"
"  }\n"
"  \n"
"  //step 2: parallel reduction:\n"
"  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (get_global_id(0) < stride)\n"
"    {\n"
"      //find the first occurring index\n"
"      if (float_buffer[get_global_id(0)] < float_buffer[get_global_id(0)+stride])\n"
"      {\n"
"        index_buffer[get_global_id(0)] = index_buffer[get_global_id(0)+stride];\n"
"        float_buffer[get_global_id(0)] = float_buffer[get_global_id(0)+stride];\n"
"      }\n"
"      \n"
"      //index_buffer[get_global_id(0)] = float_buffer[get_global_id(0)] < float_buffer[get_global_id(0)+stride] ? index_buffer[get_global_id(0)+stride] : index_buffer[get_global_id(0)];\n"
"      //float_buffer[get_global_id(0)] = max(float_buffer[get_global_id(0)], float_buffer[get_global_id(0)+stride]);\n"
"    }\n"
"  }\n"
"  \n"
"  return index_buffer[0];\n"
"}\n"
"\n"
"__kernel void index_norm_inf(\n"
"          __global float * vec,\n"
"          unsigned int size,\n"
"          __local float * float_buffer,\n"
"          __local unsigned int * index_buffer,\n"
"          global unsigned int * result) \n"
"{ \n"
"  unsigned int tmp = float_vector1_index_norm_inf_impl(vec, size, float_buffer, index_buffer);\n"
"  if (get_global_id(0) == 0) *result = tmp;\n"
"}\n"
"\n"
"\n"
; //vector_align1_index_norm_inf

const char * const vector_align1_norm_inf = 
"\n"
"////// norm_inf\n"
"float impl_norm_inf(\n"
"          __global const float * vec,\n"
"          unsigned int start_index,\n"
"          unsigned int end_index,\n"
"          __local float * tmp_buffer)\n"
"{\n"
"  float tmp = 0;\n"
"  for (unsigned int i = start_index + get_local_id(0); i < end_index; i += get_local_size(0))\n"
"    tmp = fmax(fabs(vec[i]), tmp);\n"
"  tmp_buffer[get_local_id(0)] = tmp;\n"
"  \n"
"  //step 2: parallel reduction:\n"
"  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (get_global_id(0) < stride)\n"
"      tmp_buffer[get_global_id(0)] = fmax(tmp_buffer[get_global_id(0)], tmp_buffer[get_global_id(0)+stride]);\n"
"  }\n"
"  \n"
"  return tmp_buffer[0];\n"
"}\n"
"\n"
"__kernel void norm_inf(\n"
"          __global const float * vec,\n"
"          unsigned int size,\n"
"          __local float * tmp_buffer,\n"
"          global float * group_buffer)\n"
"{\n"
"  float tmp = impl_norm_inf(vec,\n"
"                          (      get_group_id(0) * size) / get_num_groups(0),\n"
"                          ((get_group_id(0) + 1) * size) / get_num_groups(0),\n"
"                          tmp_buffer);\n"
"  \n"
"  if (get_local_id(0) == 0)\n"
"    group_buffer[get_group_id(0)] = tmp;  \n"
"}\n"
; //vector_align1_norm_inf

const char * const vector_align1_cpu_mult = 
"__kernel void cpu_mult(\n"
"          __global const float * vec,\n"
"          float factor, \n"
"          __global float * result,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec[i] * factor;\n"
"}\n"
; //vector_align1_cpu_mult

const char * const vector_align1_sub = 
"__kernel void sub(\n"
"          __global const float * vec1,\n"
"          __global const float * vec2, \n"
"          __global float * result,\n"
"          unsigned int size)\n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec1[i] - vec2[i];\n"
"}\n"
; //vector_align1_sub

const char * const vector_align1_cpu_inplace_mul_add = 
"__kernel void cpu_inplace_mul_add(\n"
"          __global float * vec1,\n"
"          __global const float * vec2,\n"
"          float factor,\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    vec1[i] += vec2[i] * factor;\n"
"}\n"
; //vector_align1_cpu_inplace_mul_add

const char * const vector_align1_sum = 
"__kernel void sum(\n"
"          __global float * vec1,\n"
"          __global float * result) \n"
"{ \n"
"  //parallel reduction on global memory:  \n"
"  \n"
"  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    if (get_global_id(0) < stride)\n"
"      vec1[get_global_id(0)] += vec1[get_global_id(0)+stride];\n"
"    barrier(CLK_GLOBAL_MEM_FENCE);\n"
"  }\n"
"  \n"
"  if (get_global_id(0) == 0)\n"
"    *result = vec1[0];  \n"
"}\n"
; //vector_align1_sum

const char * const vector_align1_inplace_mult = 
"__kernel void inplace_mult(\n"
"          __global float * vec,\n"
"          __global const float * fac, \n"
"          unsigned int size) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    vec[i] *= factor;\n"
"}\n"
; //vector_align1_inplace_mult

const char * const vector_align1_inplace_div_sub = 
"///// divide substract:\n"
"__kernel void inplace_div_sub(\n"
"          __global float * vec1,\n"
"          __global const float * vec2,\n"
"          __global const float * fac,   //CPU variant is mapped to mult_add\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    vec1[i] -= vec2[i] / factor;\n"
"}\n"
; //vector_align1_inplace_div_sub

const char * const vector_align1_norm_2 = 
"//helper:\n"
"void helper_norm2_parallel_reduction( __local float * tmp_buffer )\n"
"{\n"
"  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (get_global_id(0) < stride)\n"
"      tmp_buffer[get_global_id(0)] += tmp_buffer[get_global_id(0)+stride];\n"
"  }\n"
"}\n"
"\n"
"////// norm_2\n"
"float impl_norm_2(\n"
"          __global const float * vec,\n"
"          unsigned int start_index,\n"
"          unsigned int end_index,\n"
"          __local float * tmp_buffer)\n"
"{\n"
"  float tmp = 0;\n"
"  float vec_entry = 0;\n"
"  for (unsigned int i = start_index + get_local_id(0); i < end_index; i += get_local_size(0))\n"
"  {\n"
"    vec_entry = vec[i];\n"
"    tmp += vec_entry * vec_entry;\n"
"  }\n"
"  tmp_buffer[get_local_id(0)] = tmp;\n"
"  \n"
"  helper_norm2_parallel_reduction(tmp_buffer);\n"
"  \n"
"  return tmp_buffer[0];\n"
"};\n"
"\n"
"__kernel void norm_2(\n"
"          __global const float * vec,\n"
"          unsigned int size,\n"
"          __local float * tmp_buffer,\n"
"          global float * group_buffer)\n"
"{\n"
"  float tmp = impl_norm_2(vec,\n"
"                          (      get_group_id(0) * size) / get_num_groups(0),\n"
"                          ((get_group_id(0) + 1) * size) / get_num_groups(0),\n"
"                          tmp_buffer);\n"
"  \n"
"  if (get_local_id(0) == 0)\n"
"    group_buffer[get_group_id(0)] = tmp;  \n"
"}\n"
"\n"
; //vector_align1_norm_2

const char * const vector_align1_cpu_mul_add = 
"__kernel void cpu_mul_add(\n"
"          __global const float * vec1,\n"
"          float factor,\n"
"          __global const float * vec2,\n"
"          __global float * result,\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec1[i] * factor + vec2[i];\n"
"}\n"
; //vector_align1_cpu_mul_add

const char * const vector_align1_inplace_div_add = 
"///// divide add:\n"
"__kernel void inplace_div_add(\n"
"          __global float * vec1,\n"
"          __global const float * vec2,\n"
"          __global const float * fac,   //CPU variant is mapped to mult_add\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    vec1[i] -= vec2[i] / factor;\n"
"}\n"
; //vector_align1_inplace_div_add

const char * const vector_align1_diag_precond = 
"__kernel void diag_precond(\n"
"          __global const float * diag_A_inv, \n"
"          __global float * x, \n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    x[i] *= diag_A_inv[i];\n"
"}\n"
; //vector_align1_diag_precond

const char * const vector_align1_inplace_mul_sub = 
"__kernel void inplace_mul_sub(\n"
"          __global float * vec1,\n"
"          __global const float * vec2,\n"
"          __global const float * fac,   //CPU variant is mapped to mult_add\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    vec1[i] -= vec2[i] * factor;\n"
"}\n"
; //vector_align1_inplace_mul_sub

const char * const vector_align1_cpu_inplace_mult = 
"__kernel void cpu_inplace_mult(\n"
"          __global float * vec,\n"
"          float factor, \n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    vec[i] *= factor;\n"
"}\n"
; //vector_align1_cpu_inplace_mult

const char * const vector_align1_divide = 
"// Note: name 'div' is not allowed by the jit-compiler\n"
"__kernel void divide(\n"
"          __global const float * vec,\n"
"          __global const float * fac,  //note: CPU variant is mapped to prod_scalar\n"
"          __global float * result,\n"
"          unsigned int size)  \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec[i] / factor;\n"
"}\n"
; //vector_align1_divide

const char * const vector_align1_mul_add = 
"__kernel void mul_add(\n"
"          __global const float * vec1,\n"
"          __global const float * fac,\n"
"          __global const float * vec2,\n"
"          __global float * result,\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec1[i] * factor + vec2[i];\n"
"}\n"
; //vector_align1_mul_add

const char * const vector_align1_add = 
"__kernel void add(\n"
"          __global const float * vec1,\n"
"          __global const float * vec2, \n"
"          __global float * result,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec1[i] + vec2[i];\n"
"}\n"
; //vector_align1_add

const char * const vector_align1_inplace_divide = 
"__kernel void inplace_divide(\n"
"          __global float * vec,\n"
"          __global const float * fac,  //note: CPU variant is mapped to prod_scalar\n"
"          unsigned int size) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    vec[i] /= factor;\n"
"}\n"
; //vector_align1_inplace_divide

const char * const vector_align1_swap = 
"////// swap:\n"
"__kernel void swap(\n"
"          __global float * vec1,\n"
"          __global float * vec2,\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float tmp;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"  {\n"
"    tmp = vec2[i];\n"
"    vec2[i] = vec1[i];\n"
"    vec1[i] = tmp;\n"
"  }\n"
"}\n"
" \n"
; //vector_align1_swap

const char * const vector_align1_mul_sub = 
"///// multiply subtract:\n"
"__kernel void mul_sub(\n"
"          __global const float * vec1,\n"
"          __global const float * fac,\n"
"          __global const float * vec2,\n"
"          __global float * result,\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec1[i] * factor - vec2[i];\n"
"}\n"
; //vector_align1_mul_sub

const char * const vector_align1_vmax = 
"__kernel void vmax(\n"
"          __global float * vec1,\n"
"          __global float * result,\n"
"          unsigned int size) \n"
"{ \n"
"  //parallel reduction on global memory:\n"
"  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    if (get_global_id(0) < stride)\n"
"      vec1[get_global_id(0)] = fmax(vec1[get_global_id(0)+stride], vec1[get_global_id(0)]);\n"
"    barrier(CLK_GLOBAL_MEM_FENCE);\n"
"  }\n"
"  \n"
"  if (get_global_id(0) == 0)\n"
"    *result = vec1[0];  \n"
"}\n"
; //vector_align1_vmax

const char * const vector_align1_inner_prod = 
"//helper:\n"
"void helper_inner_prod_parallel_reduction( __local float * tmp_buffer )\n"
"{\n"
"  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (get_local_id(0) < stride)\n"
"      tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride];\n"
"  }\n"
"}\n"
"//////// inner products:\n"
"float impl_inner_prod(\n"
"          __global const float * vec1,\n"
"          __global const float * vec2,\n"
"          unsigned int start_index,\n"
"          unsigned int end_index,\n"
"          __local float * tmp_buffer)\n"
"{\n"
"  float tmp = 0;\n"
"  for (unsigned int i = start_index + get_local_id(0); i < end_index; i += get_local_size(0))\n"
"    tmp += vec1[i] * vec2[i];\n"
"  tmp_buffer[get_local_id(0)] = tmp;\n"
"  \n"
"  helper_inner_prod_parallel_reduction(tmp_buffer);\n"
"  \n"
"  return tmp_buffer[0];\n"
"}\n"
"__kernel void inner_prod(\n"
"          __global const float * vec1,\n"
"          __global const float * vec2,\n"
"          unsigned int size,\n"
"          __local float * tmp_buffer,\n"
"          global float * group_buffer)\n"
"{\n"
"  float tmp = impl_inner_prod(vec1,\n"
"                              vec2,\n"
"                              (      get_group_id(0) * size) / get_num_groups(0),\n"
"                              ((get_group_id(0) + 1) * size) / get_num_groups(0),\n"
"                              tmp_buffer);\n"
"  \n"
"  if (get_local_id(0) == 0)\n"
"    group_buffer[get_group_id(0)] = tmp;\n"
"  \n"
"}\n"
; //vector_align1_inner_prod

const char * const vector_align1_inplace_sub = 
"__kernel void inplace_sub(\n"
"          __global float * vec1,\n"
"          __global const float * vec2,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    vec1[i] -= vec2[i];\n"
"}\n"
; //vector_align1_inplace_sub

const char * const vector_align1_inplace_add = 
"__kernel void inplace_add(\n"
"          __global float * vec1,\n"
"          __global const float * vec2,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    vec1[i] += vec2[i];\n"
"}\n"
; //vector_align1_inplace_add

const char * const vector_align1_mult = 
"__kernel void mult(\n"
"          __global const float * vec,\n"
"          __global const float * fac, \n"
"          __global float * result,\n"
"          unsigned int size) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec[i] * factor;\n"
"}\n"
; //vector_align1_mult

const char * const vector_align1_sqrt_sum = 
"__kernel void sqrt_sum(\n"
"          __global float * vec1,\n"
"          __global float * result,\n"
"          unsigned int size) \n"
"{ \n"
"  //parallel reduction on global memory:  \n"
"  \n"
"  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    if (get_global_id(0) < stride)\n"
"      vec1[get_global_id(0)] += vec1[get_global_id(0)+stride];\n"
"    barrier(CLK_GLOBAL_MEM_FENCE);\n"
"  }\n"
"  \n"
"  if (get_global_id(0) == 0)\n"
"    *result = sqrt(vec1[0]);\n"
"  \n"
"}\n"
; //vector_align1_sqrt_sum

const char * const vector_align1_clear = 
"__kernel void clear(\n"
"          __global float * vec,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    vec[i] = 0;\n"
"}\n"
; //vector_align1_clear

const char * const vector_align1_norm_1 = 
"//helper:\n"
"void helper_norm1_parallel_reduction( __local float * tmp_buffer )\n"
"{\n"
"  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (get_global_id(0) < stride)\n"
"      tmp_buffer[get_global_id(0)] += tmp_buffer[get_global_id(0)+stride];\n"
"  }\n"
"}\n"
"\n"
"////// norm_1\n"
"float impl_norm_1(\n"
"          __global const float * vec,\n"
"          unsigned int start_index,\n"
"          unsigned int end_index,\n"
"          __local float * tmp_buffer)\n"
"{\n"
"  float tmp = 0;\n"
"  for (unsigned int i = start_index + get_local_id(0); i < end_index; i += get_local_size(0))\n"
"    tmp += fabs(vec[i]);\n"
"  \n"
"  tmp_buffer[get_local_id(0)] = tmp;\n"
"  \n"
"  helper_norm1_parallel_reduction(tmp_buffer);\n"
"  \n"
"  return tmp_buffer[0];\n"
"};\n"
"\n"
"__kernel void norm_1(\n"
"          __global const float * vec,\n"
"          unsigned int size,\n"
"          __local float * tmp_buffer,\n"
"          global float * group_buffer)\n"
"{\n"
"  float tmp = impl_norm_1(vec,\n"
"                          (      get_group_id(0) * size) / get_num_groups(0),\n"
"                          ((get_group_id(0) + 1) * size) / get_num_groups(0),\n"
"                          tmp_buffer);\n"
"  \n"
"  if (get_local_id(0) == 0)\n"
"    group_buffer[get_group_id(0)] = tmp;  \n"
"}\n"
"\n"
; //vector_align1_norm_1

const char * const vector_align4_inplace_mul_add = 
"__kernel void inplace_mul_add(\n"
"          __global float4 * vec1,\n"
"          __global const float4 * vec2,\n"
"          __global const float * fac,\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float factor = *fac;\n"
"  unsigned int size_div_4 = size >> 2;\n"
"  for (unsigned int i = get_global_id(0); i < size_div_4; i += get_global_size(0))\n"
"    vec1[i] += vec2[i] * factor;\n"
"}\n"
; //vector_align4_inplace_mul_add

const char * const vector_align4_cpu_inplace_mul_add = 
"__kernel void cpu_inplace_mul_add(\n"
"          __global float4 * vec1,\n"
"          __global const float4 * vec2,\n"
"          float factor,\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size/4; i += get_global_size(0))\n"
"    vec1[i] += vec2[i] * factor;\n"
"}\n"
; //vector_align4_cpu_inplace_mul_add

const char * const vector_align4_inplace_div_sub = 
"__kernel void inplace_div_sub(\n"
"          __global float4 * vec1,\n"
"          __global const float4 * vec2,\n"
"          __global const float * fac,   //CPU variant is mapped to mult_add\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size/4; i += get_global_size(0))\n"
"    vec1[i] -= vec2[i] / factor;\n"
"}\n"
; //vector_align4_inplace_div_sub

const char * const vector_align4_cpu_mul_add = 
"__kernel void cpu_mul_add(\n"
"          __global const float4 * vec1,\n"
"          float factor,\n"
"          __global const float4 * vec2,\n"
"          __global float4 * result,\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size/4; i += get_global_size(0))\n"
"    result[i] = vec1[i] * factor + vec2[i];\n"
"}\n"
; //vector_align4_cpu_mul_add

const char * const vector_align4_inplace_div_add = 
"__kernel void inplace_div_add(\n"
"          __global float4 * vec1,\n"
"          __global const float4 * vec2,\n"
"          __global const float * fac,   //CPU variant is mapped to mult_add\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size/4; i += get_global_size(0))\n"
"    vec1[i] -= vec2[i] / factor;\n"
"}\n"
; //vector_align4_inplace_div_add

const char * const vector_align4_inplace_mul_sub = 
"__kernel void inplace_mul_sub(\n"
"          __global float4 * vec1,\n"
"          __global const float4 * vec2,\n"
"          __global const float * fac,   //CPU variant is mapped to mult_add\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size/4; i += get_global_size(0))\n"
"    vec1[i] -= vec2[i] * factor;\n"
"}\n"
; //vector_align4_inplace_mul_sub

const char * const vector_align4_mul_add = 
"__kernel void mul_add(\n"
"          __global const float4 * vec1,\n"
"          __global const float * fac,\n"
"          __global const float4 * vec2,\n"
"          __global float4 * result,\n"
"          unsigned int size\n"
"          ) \n"
"{ \n"
"  float factor = *fac;\n"
"  for (unsigned int i = get_global_id(0); i < size/4; i += get_global_size(0))\n"
"    result[i] = vec1[i] * factor + vec2[i];\n"
"}\n"
; //vector_align4_mul_add

  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif
