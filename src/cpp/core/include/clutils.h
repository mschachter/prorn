
#ifndef CLUTILS_H
#define CLUTILS_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <map>
#include <string>

#include <CL/cl.h>

using namespace std;

string read_file(string filePath);

void print_platform_info(cl_platform_id pid);
void print_device_info(cl_device_id did);

cl_program* create_program_from_string(cl_context context, string programString);
cl_program* create_program_from_file(cl_context context, string filePath);

cl_kernel* create_kernel(cl_program prog, string kernelName);

typedef map<string, cl_program*> CLProgramMap;

class CLDevice {

public:
   CLDevice();
   ~CLDevice();

   void init();

   cl_platform_id platformId;
   cl_uint numPlatforms;
   cl_device_id deviceId;
   cl_uint numDevices;
   cl_context context;
	cl_command_queue commandQueue;

   cl_kernel* createKernel(string fileName, string kernelName);

protected:
   CLProgramMap programs;
      

};


#endif
