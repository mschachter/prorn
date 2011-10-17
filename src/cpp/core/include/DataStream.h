#ifndef DATASTREAM_H
#define DATASTREAM_H

#include <opencv/cv.h>

using namespace cv;


class DataStream {

public:

   virtual Mat* poll(bool refresh) = 0;
   virtual int getNDim() = 0;
   virtual int* getDims() = 0;

};

class RandomStreamInt : public DataStream {

public:
   
   RandomStreamInt(int nrows, int ncols, int mean, int std);
   ~RandomStreamInt();

   Mat* poll(bool refresh);
   int getNDim();
   int* getDims();

protected:      
   Mat* img;
   int* dims;
   int ndim;

   int mean;
   int std;
};

#endif


