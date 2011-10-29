

QMAKE4 = qmake4
PROJECTS = core ui

all: 
	cd core; make all;
	cd ui; $(QMAKE4); rm -f ui; make all;


clean:
	cd core; make clean;
	cd ui; make clean;
	cd sandbox; make clean;

