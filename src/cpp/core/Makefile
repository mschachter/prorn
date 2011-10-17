
CC=g++
INC = -Iinclude -I/usr/local/include -I/usr/include
LIB = -lOpenCL -lopencv_core
CFLAGS = -c
AR = ar
ARFLAGS = -rcsv

LIBFILE = libneurogoats.a

OBJECTS = Network.o DataStream.o StreamTransformer.o clutils.o

all: ng

ng: $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIBFILE) $(OBJECTS)
	ranlib $(LIBFILE)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INC) $(LIB) -o $@ $<

clean:
	rm -rf *.o $(LIBFILE)

