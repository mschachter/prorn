#include <stdio.h>

#include "DataStream.h"

RandomStreamInt::RandomStreamInt(int nrows, int ncols, int meanVal, int stdVal)
{
   //printf("Creating random int stream, rows=%d, cols=%d, mean=%d, std=%d\n", nrows, ncols, meanVal, stdVal);
   img = new Mat(Size(ncols, nrows), CV_8UC1);

   ndim = 2;
   dims = new int[2];
   dims[0] = nrows;
   dims[1] = ncols;

   mean = meanVal;
   std = stdVal;
}

RandomStreamInt::~RandomStreamInt()
{
   delete img;
   delete dims;
}

Mat* RandomStreamInt::poll(bool refresh)
{
   if (refresh) {
      printf("Randomizing image for poll(), rows=%d, cols=%d, mean=%d, std=%d\n", img->rows, img->cols, mean, std);   
      randn(*img, Scalar::all(mean), Scalar::all(std));
      /*
      for (int k = 0; k < img->rows; k++) {
         uchar* row = img->ptr<uchar>(k);
         for (int j = 0; j < img->cols; j++) {
            printf("img[%d][%d]=%d\n", k, j, row[j]);
         }
      }
      */
   }

   return img;
}

int RandomStreamInt::getNDim()
{
   return ndim;
}


int* RandomStreamInt::getDims()
{
   return dims;
}

