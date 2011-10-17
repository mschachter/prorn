__kernel void step(__global float** state, __global float** params, __global unsigned int* unitIndex,
                   __global float* weights, __global unsigned int* weightIndex, __global usigned int** weightIndexRange,
                   __global float* prevOutput, __global float* newOutput,
                   __global float stepSize)
{
	const uint gpuId = get_global_id(0);
   const uint unitId = unitIdex[gpuIndx];
   
   float gmem = params[gpuId][0];    //membrane conductance
   float tau = params[gpuId][1];     //time constant
   float thresh = params[gpuId][3];  //threshold
   float synTau = params[gpuId][4];  //synaptic decay time constant

   int windx1 = weightIndexRange[gpuId][0];
   int windx2 = weightIndexRange[gpuId][1];   
   int nwts = windx2 - windx1 + 1;   
   
   //compute input current
   uint k, preId;
   float wval;
   float isyn = 0.0;
   for (k = 0; k < nwts; k++) {
      wval = weights[windx1+k];
      preId = weightIndex[windx1+k];
      isyn += wval * prevOutput[ preId ];
   }

   //update state
   float vnew = state[unitId][0] + stepSize*((-state[unitId][0] + (isyn / gmem)) / tau);
   state[unitId][0] = vnew;

   //update output
   if (vnew > thresh) {
      newOutput[unitId] = 1.0;
   } else {
      if (newOutput > 0.0) {
         newOutput[unitId] = prevOutput[i] + stepSize*(-prevOutput[unitId] / synTau);
      } else {
         newOutput[unitId] = 0.0;
      }      
   }
}
