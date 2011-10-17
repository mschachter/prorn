#ifndef STREAMPROJECTION_H
#define STREAMPROJECTION_H

#include "DataStream.h"

class StreamTransformer {

public:
   virtual void setDataStream(DataStream* ds) = 0;
   virtual void* transform() = 0;
};

class StreamTransformer2d : public StreamTransformer {

public:
   StreamTransformer2d(DataStream* ds);
   ~StreamTransformer2d();

   void setDataStream(DataStream* ds);
   void* transform();

protected:
   DataStream* dataStream;
};

#endif
