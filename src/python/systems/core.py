'''
Created on Sep 26, 2009

@author: mschachter
'''

import numpy as np

class System:
    def __init__(self, initialState, params = None):
        self.initialState = initialState
        self.parameters = params
        self.stateShape = initialState.shape
        self.name = 'Dynamical System'


class ContinuousSystem(System):
    pass


class DiscreteSystem(System):
    pass


class SystemInspector:
    def __init__(self, system):
        pass

    def getNullclines(self):
        pass
    
    def getFixedPoints(self):
        pass        


class LotkaVolterraSystem(ContinuousSystem):    
    
    def __init__(self, initialState = np.array([0.0, 0.0])):
        System.__init__(self, initialState, self.rhs)
        self.name = 'Lotka-Volterra'
    
    def rhs(self, x, t = None):
        #print '[LotkaVolterraSystem.rhs()]'
        x1 = x[0]*(3-x[0]-2*x[1])
        x2 = x[1]*(2-x[0]-x[1]) 
        return np.array( [ x1, x2 ] )
    
