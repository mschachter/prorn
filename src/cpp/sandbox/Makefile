
COREDIR=../core

CL_INCLUDE_DIR=/opt/AMDAPP/include
CL_LIB_DIR=/opt/AMDAPP/lib/x86_64

CC=g++
#INC = -Iinclude -I/usr/local/include -I/usr/include -I$(CL_INCLUDE_DIR) -I$(COREDIR)/include
INC = -I$(CL_INCLUDE_DIR)
#LIB = -lGL -lglut -lopencv_core -L$(COREDIR) -lneurogoats -lOpenCL
#LIB = -L$(COREDIR) -L$(CL_LIB_DIR) -lneurogoats -lOpenCL
LIB = -L$(CL_LIB_DIR) -lOpenCL
CFLAGS = -c

all: cltest

cltest: cltest.cpp
	$(CC) -c -o clutils.o $(INC) $(LIB) clutils.cpp
	$(CC) -o cltest clutils.o $(INC) $(LIB) cltest.cpp

clean:
	rm -rf *.o gltest cltest testptrs

