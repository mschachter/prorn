#include <stdio.h>

#include <CL/cl.h>

#include <vector>

using namespace std;

int main(int argc, char* argv[])
{
   /*
   int n = 10;
   int* a = new int[n];
   for (int k = 0; k < n; k++) {
      a[k] = (k+1)*3;
   }

   printf("Re-referencing a...\n");
   int* b = new int[n];
   for (int k = 0; k < n; k++) {
      b[k] = a[k];
   }

   printf("Printing out...\n");
   for (int k = 0; k < n; k++) {
      printf("a[%d]=%d, b[%d]=%d\n", k, a[k], k, b[k]);
   }
   */

   vector<int>* tv = new vector<int>[10];
   tv[0].push_back(12);
   tv[1].push_back(25);
   tv[1].push_back(33);

   printf("tv[0].size()=%d, tv[1].size()=%d, tv[2].size()=%d, tv[9].size()=%d\n",
          tv[0].size(), tv[1].size(), tv[2].size(), tv[9].size());

}
