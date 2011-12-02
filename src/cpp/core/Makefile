
CL_INCLUDE_DIR=/opt/AMDAPP/include
CL_LIB_DIR=/opt/AMDAPP/lib/x86_64

CC=g++
INC = -Iinclude -I/usr/local/include -I/usr/include -I$(CL_INCLUDE_DIR)
LIB = -L$(CL_LIB_DIR) -lOpenCL 
CFLAGS = -c -fPIC
AR = ar
ARFLAGS = -rcsv

PROJECT_NAME=prorn
LIBNAME = lib$(PROJECT_NAME)
LIBFILE = $(LIBNAME).a
SOFILE =  $(LIBNAME).so

OBJECTS = Network.o clutils.o

all: $(PROJECT_NAME)

$(PROJECT_NAME): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIBFILE) $(OBJECTS)
	ranlib $(LIBFILE)
	$(CC) -shared $(LIB) -o $(SOFILE) $(OBJECTS)
	

%.o: %.cpp
	$(CC) $(CFLAGS) $(INC) $(LIB) -o $@ $<

clean:
	rm -rf *.o $(LIBFILE) $(SOFILE)

