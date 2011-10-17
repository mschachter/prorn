
#include "StreamTransformer.h"

#include <stdio.h>

StreamTransformer2d::StreamTransformer2d(DataStream* ds)
{
   if (ds->getNDim() != 2) {
      printf("Cannot create StreamTransformer2d, data stream is not 2-dimensional!\n");
      return;
   }

   dataStream = ds;
}

StreamTransformer2d::~StreamTransformer2d()
{

}

void StreamTransformer2d::setDataStream(DataStream* ds)
{
   dataStream = ds;
}

void* StreamTransformer2d::transform()
{
   Mat* img = dataStream->poll(false);
   return img->data;
}

