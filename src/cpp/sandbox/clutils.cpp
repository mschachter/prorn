
#include "clutils.h"

string read_file(string filePath)
{
   string fileTxt;

   string ln;
   ifstream f(filePath.c_str());
   if (f.is_open()) {
      while (f.good()) {
         getline(f, ln);
         fileTxt.append(ln);
      }
      f.close();
   }   

   return fileTxt;
}

cl_program* create_program_from_file(cl_context context, string filePath)
{
   cl_program* prog;   
   size_t flen;
   string fstr = read_file(filePath);   
   prog = create_program_from_string(context, fstr);
   return prog;
}

cl_program* create_program_from_string(cl_context context, string programString)
{
   cl_int rval;
   cl_program* prog = new cl_program;

   const char *constSrc[] = {programString.c_str()}; 
   size_t strLen = programString.size();
   *prog = clCreateProgramWithSource(context, 1, constSrc, &strLen, &rval);

	rval = clBuildProgram(*prog, 0, NULL, "", NULL, NULL);
	if(rval != CL_SUCCESS) {
      cerr << "Compile error for program:" << endl << programString << endl;
      delete prog;
      prog = NULL;
	}
   
   return prog;
}


cl_kernel* create_kernel(cl_program prog, string kernelName)
{
   cl_int rval;
   cl_kernel* kernel = new cl_kernel;
   *kernel = clCreateKernel(prog, kernelName.c_str(), &rval);
   if (rval != CL_SUCCESS) {
      cerr << "Could not create kernel: " << kernelName << endl;
      delete kernel;
      kernel = NULL;
   }
   
   return kernel;
}

void print_device_info(cl_device_id did)
{
   cl_int rval;
   size_t strSize = 255;
   char* pinfo = new char[strSize];
   
   size_t aSize;
   rval = clGetDeviceInfo(did, CL_DEVICE_NAME, sizeof(char)*strSize, pinfo, &aSize);
   pinfo[aSize] = '\0';
   if (rval != CL_SUCCESS) {
      printf("Error obtaining device !\n");
      return;
   }
   printf("\tDevice Name: %s\n", pinfo);

   delete pinfo;
} 

void print_platform_info(cl_platform_id pid)
{
   cl_int rval;
   size_t strSize = 255;
   char* pinfo = new char[strSize];
   
   size_t aSize;
   rval = clGetPlatformInfo(pid, CL_PLATFORM_PROFILE, sizeof(char)*strSize, pinfo, &aSize);
   pinfo[aSize] = '\0';
   if (rval != CL_SUCCESS) {
      printf("Error obtaining platform profile!\n");
      return;
   }
   printf("\tPlatform Profile: %s\n", pinfo);

   rval = clGetPlatformInfo(pid, CL_PLATFORM_NAME, sizeof(char)*strSize, pinfo, &aSize);
   if (rval != CL_SUCCESS) {
      printf("Error obtaining platform name!\n");
      return;
   }
   pinfo[aSize] = '\0';
   printf("\tPlatform Name: %s\n", pinfo);

   rval = clGetPlatformInfo(pid, CL_PLATFORM_VERSION, sizeof(char)*strSize, pinfo, &aSize);
   if (rval != CL_SUCCESS) {
      printf("Error obtaining platform version!\n");
      return;
   }
   pinfo[aSize] = '\0';
   printf("\tPlatform Version: %s\n", pinfo);
   
   rval = clGetPlatformInfo(pid, CL_PLATFORM_VENDOR, sizeof(char)*strSize, pinfo, &aSize);
   if (rval != CL_SUCCESS) {
      printf("Error obtaining platform vendor!\n");
      return;
   }
   pinfo[aSize] = '\0';
   printf("\tPlatform Vendor: %s\n", pinfo);

   rval = clGetPlatformInfo(pid, CL_PLATFORM_EXTENSIONS, sizeof(char)*strSize, pinfo, &aSize);
   if (rval != CL_SUCCESS) {
      printf("Error obtaining platform extensions!\n");
      return;
   }
   pinfo[aSize] = '\0';
   printf("\tPlatform Extensions: %s\n", pinfo);

   delete pinfo;   
}


CLDevice::CLDevice()
{
   
}

CLDevice::~CLDevice()
{
   
}

void CLDevice::init()
{
   cl_int error;
      
   printf("[CLDevice] Getting platform IDs...");
	error = clGetPlatformIDs(1, &platformId, &numPlatforms);
   printf(" got %d platforms:\n", numPlatforms);
   print_platform_info(platformId);
   
   printf("[CLDevice] Getting device IDs...");
	error = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, &numDevices);
   printf(" got %d devices:\n", numDevices);
   print_device_info(deviceId);

	cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platformId, 0};
	context = clCreateContext(properties, 1, &deviceId, NULL, NULL, &error);
   printf("[CLDevice] GCreated context...\n");
   
   commandQueue = clCreateCommandQueue(context, deviceId, 0, &error);
   printf("[CLDevice] GCreated command queue...\n");
}


cl_kernel* CLDevice::createKernel(string fileName, string kernelName)
{
   CLProgramMap::iterator it = programs.find(fileName);
   cl_program* prog;
   if (it == programs.end()) {
      prog = create_program_from_file(context, fileName);
      programs[fileName] = prog;
   }
   prog = programs[fileName];
   cl_kernel* ker = NULL;
   if (prog != NULL) {
      ker = create_kernel(*prog, kernelName);  
   }

   return ker;
}

