
CC=g++
INC = -Iinclude -I/usr/local/include -I/usr/include
LIB = -lOpenCL
CFLAGS = -c -fPIC
AR = ar
ARFLAGS = -rcsv

LIBNAME = libneurogoats
LIBFILE = $(LIBNAME).a
SOFILE =  $(LIBNAME).so.1.0

OBJECTS = Network.o clutils.o

all: ng

ng: $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIBFILE) $(OBJECTS)
	ranlib $(LIBFILE)
	$(CC) -shared $(LIB) -o $(SOFILE) $(OBJECTS)
	

%.o: %.cpp
	$(CC) $(CFLAGS) $(INC) $(LIB) -o $@ $<

clean:
	rm -rf *.o $(LIBFILE) $(SOFILE)

