import numpy as np
import matplotlib.pyplot as plt

class Bounds:
    pass

class Grid:
    pass

class PhasePlot:
    
    def __init__(self, sys, bounds, spacing = None):
        self.system = sys
        self.bounds = bounds
        if spacing is None:
            spacing = .1
        self.spacing = spacing
        self.grid = Grid()
        
        self.constructGrid()
        
        
    def constructGrid(self):
        if self.system.stateShape == (2,):
            X, Y = np.meshgrid(np.arange(self.bounds.xmin, self.bounds.xmax, self.spacing),
                               np.arange(self.bounds.ymin, self.bounds.ymax, self.spacing))
            print 'X.shape=',X.shape
            print 'Y.shape=',Y.shape

            U = np.zeros(X.shape)
            V = np.zeros(Y.shape)
            for i in range(X.shape[0]):
                for j in range(Y.shape[1]):
                    x = X[i, j]
                    y = Y[i, j]
                    state = np.array([x, y])
                    dstate = self.system.rhs(state)
                    #dstate /= np.linalg.norm(dstate)
                    U[i, j] = dstate[0]
                    V[i, j] = dstate[1]

            self.X = X
            self.Y = Y
            self.U = U
            self.V = V

    def renderPlot(self):
        plt.figure()
        Q = plt.quiver(self.X, self.Y, self.U, self.V)

