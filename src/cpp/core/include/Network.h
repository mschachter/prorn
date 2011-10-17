
#include <string>
#include <vector>
#include <map>

#include <opencv/cv.h>

#include "clutils.h"
#include "DataStream.h"

using namespace cv;
using namespace std;

typedef unsigned int UnitId;
typedef unsigned int WeightId;   //should be a long long
typedef unsigned int LayerId;

class Unit {
   
public:
   UnitId id;
   float* initialState;
   float* state;
   float* params;
   int ndim;
   int nparams;
   bool external;
   string* clFileName;
};

typedef map<UnitId, Unit*> UnitMap;

class Layer {
public:
   LayerId id;
   UnitMap units;
   virtual bool isExternal() = 0;
};

class ExternalLayer : public Layer {
public:
   virtual void update() = 0;
   virtual float getOutput(UnitId uid);
};

class DataStreamLayer : public ExternalLayer {

public:
   DataStreamLayer(DataStream* ds);
   ~DataStreamLayer();

   bool isExternal();   
   void update();
};

class IFLayer : public Layer {
public:
   bool isExternal();
};

typedef map<LayerId, Layer*> LayerMap;

class Network {

public:

   Network();
   ~Network();

   UnitId addUnit(Unit* u);
   void connect(UnitId u1, UnitId u2, float weight);
   void build();
   void step();

   LayerId addLayer(Layer* l);

   /* Non-GPU member variables */
   vector<Unit*> units;
   SparseMat* weights;   
   LayerMap layers;  
   unsigned int numNonExUnits;
   float totalMem;
   unsigned int numConns;
   unsigned int numStates;
   unsigned int numParams;
   UnitId* kernelIndexArr; //maps index in global kernel array to unit ID

   cl_kernel* clKernel;
   CLDevice clDevice;

   /* Stuff copied to GPU */
   float** state;             //the state of non-external units
   float** params;            //parameters of non-external units
   UnitId* unitIndexArr;      //maps index in state/params to UnitId
   float* weightArr;          //weights for non-external units, index range found in weightIndexRange
   UnitId* weightIndex;       //same length as weightArr, weightArr contains actual weight, this array contains UnitId of pre-synaptic connection
   unsigned int** weightIndexRange; //range of weights in weightArr for non-external units, Nx2 array, N=# of non-external units   
   float* output;             //output of all units, of length units.size()
   float* tempOutput;         //output of all units, of length units.size()
   float stepSize;

   cl_mem clState;
   cl_mem clParams;
   cl_mem clUnitIndex;
   cl_mem clWeights;
   cl_mem clWeightIndex;
   cl_mem clWeightIndexRange;
   cl_mem clPrevOutput;
   cl_mem clNewOutput;
   cl_mem clStepSize;
      
protected:

   bool built;
   
   UnitId getNextUnusedId();

   void initWeights();
   void buildWeightData();
   void buildStateData();
   void buildKernels();
   void allocateMemory();
   void prepareKernel();
};

