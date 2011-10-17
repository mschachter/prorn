#include "Network.h"

Network::Network()
{
   weights = NULL;  
   numNonExUnits = 0;
   totalMem = 0.0;
   numConns = 0;
   numStates = 0;
   numParams = 0;
   kernelIndexArr = NULL;

   state = NULL;
   params = NULL;
   unitIndexArr = NULL;
   weightArr = NULL;
   weightIndex = NULL;
   weightIndexRange = NULL;
   output = NULL;
   tempOutput = NULL;
   stepSize = 0.0005;

   built = false;
}

Network::~Network()
{
   delete weights;
   delete kernelIndexArr;

   delete state;
   delete params;
   delete unitIndexArr;
   delete weightArr;
   delete weightIndex;
   delete weightIndexRange;
   delete output;
   delete tempOutput;
}

UnitId Network::getNextUnusedId()
{
   return units.size();
}

void Network::step()
{
   if (!built) build();

   if (built) {
      LayerMap::iterator it;
      Layer* l;
      //update the external layers
      for (it = layers.begin(); it != layers.end(); it++) {         
         l = it->second;
         if (l->isExternal()) {
            //update layer and add outputs to native outputs array
            ExternalLayer* el = (ExternalLayer*) l;
            el->update();
            UnitMap::iterator uit;
            UnitId unitId;
            for (uit = el->units.begin(); uit != el->units.end(); uit++) {
               output[unitId] = el->getOutput(unitId);
            }
         }
      }

      //run the kernel
      cl_int err;
      size_t workSize = (size_t) numNonExUnits;
      prepareKernel();
      err = clEnqueueNDRangeKernel(clDevice.commandQueue, *clKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
	   err = clFinish(clDevice.commandQueue);
      if (err != CL_SUCCESS) {
         printf("Failed to execute kernel!\n");
      }

      //copy the new output to the previous output
      clEnqueueCopyBuffer(clDevice.commandQueue, clNewOutput, clPrevOutput, 0, 0, sizeof(float)*units.size(), 0, NULL, NULL);
      err = clFinish(clDevice.commandQueue);
   }
}


UnitId Network::addUnit(Unit* u)
{
   UnitId id = getNextUnusedId();
   u->id = id;
   units.push_back(u);
   return id;
}

LayerId Network::addLayer(Layer* l)
{
   LayerId nextId = layers.size();
   layers[nextId] = l;

   for (int k = 0; k < l->units.size(); k++) {
      Unit* u = l->units[k];
      addUnit(u);
   }

   return nextId;
}

void Network::connect(UnitId u1, UnitId u2, float weight)
{
   if (weights == NULL) initWeights();

   int idx[] = {u1, u2};
   weights->ref<float>(idx) = weight;
}

void Network::initWeights()
{
   printf("Initializing weight matrix, don't add any more units!\n");
   if (weights != NULL) delete weights;
   
   int sz[] = {units.size(), units.size()};
   weights = new SparseMat(2, sz, CV_32F); 
}

void Network::build()
{
   built = false;
   if (units.size() > 0) {
      buildStateData();
      buildWeightData();
      buildKernels();
      allocateMemory();

      totalMem = (sizeof(float)*numStates +      
                  sizeof(float)*numParams + 
                  sizeof(unsigned int)*numNonExUnits +                  
                  sizeof(unsigned int)*numConns*3 + 
                  sizeof(float)*numConns +
                  sizeof(float)*units.size()*2) / (1024*1024);

      printf("Network built:\n");
      printf("  # of Units: %d\n", units.size());
      printf("  # of Params: %d\n", numParams);
      printf("  # of Connections: %d", numConns);
      printf("  # of States: %d\n", numStates);
      printf("   Total GPU Memory: %fMB\n", totalMem);


      built = true;
   } else {
      printf("Cannot build network, no units have been added!");
   }
}

void Network::buildStateData()
{
   //initialize index for non-external units
   if (unitIndexArr != NULL) delete unitIndexArr;
   numNonExUnits = 0;
   UnitId k;
   //count the # of non-external units
   for (k = 0; k < units.size(); k++) {
      if (!units[k]->external) numNonExUnits++;
   }
   //initialize mapping between unit id and kernel index
   kernelIndexArr = new UnitId[units.size()];
   for (k = 0; k < units.size(); k++) {
      kernelIndexArr[k] = 0;  //potentially bug-causing...
   }
   //create mapping between kernel index and unit id
   unitIndexArr = new UnitId[numNonExUnits];
   UnitId nonExIndx = 0;
   for (k = 0; k < units.size(); k++) {
      if (!units[k]->external) {
         unitIndexArr[nonExIndx] = k;
         kernelIndexArr[k] = nonExIndx;
         nonExIndx++;
      }      
   }   


   //initialize state and params vector
   if (state != NULL) delete state;
   state = new float*[numNonExUnits];
   params = new float*[numNonExUnits];
   int ndim, m;
   UnitId uindx;
   for (k = 0; k < numNonExUnits; k++) {
      //initialize state
      uindx = unitIndexArr[k];
      ndim = units[uindx]->ndim;
      state[k] = new float[ndim];
      for (m = 0; m < ndim; m++) {
         state[k][m] = units[uindx]->initialState[m];
         numStates++;
      }
      //initialize params
      ndim = units[uindx]->nparams;
      for (m = 0; m < ndim; m++) {
         params[k][m] = units[uindx]->params[m];   
         numParams++;
      }
   }

   //initialize output vectors
   if (output != NULL) delete output;
   output = new float[units.size()];
   if (tempOutput != NULL) delete tempOutput;
   tempOutput = new float[units.size()];
   for (k = 0; k < units.size(); k++) {
      output[k] = 0.0f;
      tempOutput[k] = 0.0f;
   }
}


void Network::buildWeightData()
{
   if (weights == NULL) initWeights();
   
   //construct the weightIndicies array, each element of
   //which contains a vector of pre-synaptic connections
   //for each a given unit
   vector<UnitId>* weightIndicies = new vector<UnitId>[units.size()];   
   numConns = 0;
   SparseMatConstIterator it = weights->begin(), it_end = weights->end();
   for(; it != it_end; ++it) {
      const SparseMat::Node* anode = it.node();
      const int* indx = anode->idx;
      float wval = weights->value<float>(anode->idx);
      int pre = indx[0];
      int post = indx[1];      
      printf("buildWeightData: pre=%d, post=%d, wval=%f\n", pre, post, wval);
      
      //add to weight index
      weightIndicies[post].push_back((UnitId) pre);
      numConns++;
   }

   //construct weightArr and weightIndexRange
   if (weightArr != NULL) delete weightArr;
   weightArr = new float[numConns];
   if (weightIndex != NULL) delete weightIndex;
   weightIndex = new UnitId[numConns];
   if (weightIndexRange != NULL) delete weightIndexRange;
   weightIndexRange = new unsigned int*[numNonExUnits];
   UnitId k;
   for (k = 0; k < numNonExUnits; k++)  weightIndexRange[k] = NULL;

   unsigned int cindx, gpuIndx, nwts, m;   
   int windx[2];
   cindx = 0;
   for (k = 0; k < units.size(); k++) {
      nwts = weightIndicies[k].size();
      if (nwts > 0) {
         gpuIndx = kernelIndexArr[k];
         weightIndexRange[gpuIndx] = new unsigned int[2];
         weightIndexRange[gpuIndx][0] = cindx;
         weightIndexRange[gpuIndx][1] = cindx + nwts - 1;
         for (m = 0; m < nwts; m++) {
            weightIndex[cindx] = weightIndicies[k][m];
            windx[0] = m;
            windx[1] = k;
            weightArr[cindx] = weights->value<float>(windx);
            cindx++;
         }
      }
   }

   //throw out weightIndicies
   delete weightIndicies;
}


void Network::buildKernels()
{
   map<string*, bool> kmap;
   map<string*, bool>::iterator it;
   for (int k = 0; k < units.size(); k++) {
      it = kmap.find(units[k]->clFileName);
      if (it == kmap.end()) {
         kmap[units[k]->clFileName] = true;
      }   
   }
   if (kmap.size() > 1) {
      printf("Network doesn't support multiple kernels yet!\n");
   }
   it = kmap.begin();
   string* clFileName = it->first;
   
   clDevice.init();
   clKernel = clDevice.createKernel(*clFileName, string("step"));
}

void Network::allocateMemory()
{
   cl_int err;
   clState = clCreateBuffer(clDevice.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)*numStates, state, &err);   
   clParams = clCreateBuffer(clDevice.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)*numParams, params, &err);
   clUnitIndex = clCreateBuffer(clDevice.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(UnitId)*numNonExUnits, unitIndexArr, &err);   
   clWeights = clCreateBuffer(clDevice.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)*numConns, weightArr, &err);
   clWeightIndex = clCreateBuffer(clDevice.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int)*numConns, weightIndex, &err);
   clWeightIndexRange = clCreateBuffer(clDevice.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int)*numConns*2, weightIndexRange, &err);
   clPrevOutput = clCreateBuffer(clDevice.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)*units.size(), output, &err);
   clNewOutput = clCreateBuffer(clDevice.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)*units.size(), tempOutput, &err);
   clStepSize = clCreateBuffer(clDevice.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), &stepSize, &err);
}

void Network::prepareKernel()
{
   //set up the kernel
   clSetKernelArg(*clKernel, 0, sizeof(cl_mem), clState);
   clSetKernelArg(*clKernel, 1, sizeof(cl_mem), clParams);
   clSetKernelArg(*clKernel, 2, sizeof(cl_mem), clUnitIndex);
   clSetKernelArg(*clKernel, 3, sizeof(cl_mem), clWeights);
   clSetKernelArg(*clKernel, 4, sizeof(cl_mem), clWeightIndex);
   clSetKernelArg(*clKernel, 5, sizeof(cl_mem), clWeightIndexRange);
   clSetKernelArg(*clKernel, 6, sizeof(cl_mem), clPrevOutput);
   clSetKernelArg(*clKernel, 7, sizeof(cl_mem), clNewOutput);
   clSetKernelArg(*clKernel, 8, sizeof(cl_mem), &stepSize);
}

