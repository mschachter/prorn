
#include <stdio.h>
#include "clutils.h"

int main(int nargs, char** args)
{
   cl_int error;
   CLDevice clDevice;
   clDevice.init();

   string filePath("ng.cl");
   string kernelName("zero_mem");
   
   cl_kernel* zmem = clDevice.createKernel(filePath, kernelName);

   size_t numNodes = 20;

   float* v_orig = new float[numNodes];
   for (int k = 0; k < numNodes; k++) {
      v_orig[k] = 3.4;
   }

   printf("Allocating GPU memory...\n");
   //cl_mem v_dev = clCreateBuffer(clDevice.context, CL_MEM_READ_WRITE, sizeof(float)*numNodes, NULL, &error);
   cl_mem v_dev = clCreateBuffer(clDevice.context, CL_MEM_COPY_HOST_PTR, sizeof(float)*numNodes, v_orig, &error);

   printf("Executing kernel...\n");
   clSetKernelArg(*zmem, 0, sizeof(v_dev), &v_dev);   
	error = clEnqueueNDRangeKernel(clDevice.commandQueue, *zmem, 1, NULL, &numNodes, NULL, 0, NULL, NULL);
	error = clFinish(clDevice.commandQueue);
   if (error != CL_SUCCESS) {
      printf("Failed to execute kernel!\n");
      return 1;
   }
   
   printf("Reading results from GPU...\n");
   float* v = new float[numNodes];
   error = clEnqueueReadBuffer(clDevice.commandQueue, v_dev, CL_TRUE, 0, sizeof(float)*numNodes, v, 0, NULL, NULL);   

   for (int k = 0; k < numNodes; k++) {
      printf("v[%d] = %f\n", k, v[k]);
   }

   return 0;
}

