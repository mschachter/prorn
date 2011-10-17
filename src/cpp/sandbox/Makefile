
COREDIR=../core

CC=g++
INC = -Iinclude -I$(COREDIR)/include -I/usr/local/include -I/usr/include
LIB = -lGL -lglut -lopencv_core -L$(COREDIR) -lneurogoats -lOpenCL
CFLAGS = -c

all: gltest testptrs ng

gltest: gltest.cpp
	$(CC) -o gltest $(INC) $(LIB) gltest.cpp

testptrs: testptrs.cpp
	$(CC) -o testptrs $(INC) $(LIB) testptrs.cpp

ng: ng.cpp
	$(CC) -o ng $(INC) $(LIB) ng.cpp $(COREDIR)/clutils.o

clean:
	rm -rf *.o gltest ng testptrs

