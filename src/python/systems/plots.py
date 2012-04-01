'''
Created on Oct 1, 2009

@author: mschachter
'''
from pylab import *
from numpy.core import array, arange, zeros
from numpy.linalg import norm

class Bounds:
    pass

class Grid:
    pass

class PhasePlot:
    
    def __init__(self, sys, bounds, spacing = None):
        self.system = sys;
        self.bounds = bounds;
        if (spacing == None):
            spacing = .1
        self.spacing = spacing
        self.grid = Grid()
        
        self.constructGrid()
        
        
    def constructGrid(self):
        if self.system.stateShape == (2,):            
            self.grid.x,self.grid.y = meshgrid(arange(self.bounds.xmin, self.bounds.xmax, self.spacing),
                                               arange(self.bounds.ymin, self.bounds.ymax, self.spacing));
        
        xvals = zeros( (self.grid.x.shape[0], self.grid.x.shape[1]) )
        yvals = zeros( (self.grid.x.shape[0], self.grid.x.shape[1]) )
        
        aclrs = zeros( (self.grid.x.shape[0], self.grid.x.shape[1]) )        
        
        for xindx in range(0,self.grid.x.shape[0]):
            for yindx in range(0,self.grid.x.shape[1]):
                
                xpnt = self.grid.x[xindx,yindx]
                ypnt = self.grid.y[xindx,yindx]                
                s = array([xpnt,ypnt])
                sval = self.system.valueAt(s)
                snorm = norm(sval)
                sval /= snorm
                
                aclrs[xindx,yindx] = snorm                                       
                xvals[xindx,yindx] = sval[0]
                yvals[xindx,yindx] = sval[1]
                
                #print '({0},{1})=({2},{3})'.format(xpnt, ypnt, sval[0], sval[1])

        self.xvals = xvals
        self.yvals = yvals
        self.aclrs = aclrs;                
        
    def renderPlot(self):
        
        figure()                
        Q = quiver(self.grid.x, self.grid.y, self.xvals, self.yvals, self.aclrs, scale=30)        
        plot(self.grid.x, self.grid.y, 'ko', markersize=0.75)                
        axis([self.bounds.xmin, self.bounds.xmax, self.bounds.ymin, self.bounds.ymax])        
        colorbar()
        title(self.system.name)
        show()
