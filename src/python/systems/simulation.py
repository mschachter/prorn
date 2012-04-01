'''
Created on Oct 12, 2009

@author: mschachter
'''

from numpy.linalg import norm
from scipy.optimize import fmin,fsolve

class Simulator:
    
    def __init__(self, sys, timestep):
        self.system = sys
        self.stepSize = timestep
        self.time = 0;        
        self.solvers = {'ForwardEuler': ForwardEuler(), 'BackwardEuler': BackwardEuler(), 'RungeKutta4': RungeKutta4()}
        self.currentValue = self.system.initialState
                        
        
    def next(self, solverName = 'ForwardEuler', stepSize = None):
        
        dt = self.stepSize
        if (stepSize != None):
            dt = stepSize
        
        solver = self.solvers[solverName]
        self.previousValue = self.currentValue        
        self.currentValue = solver.step(self.system.stepFunction, self.currentValue, self.time, dt)
        self.time += dt
        #print '[Simulator.next] time={0}:\n\tpreviousValue={1}\n\tcurrentValue={2}'.format(self.time, self.previousValue, self.currentValue)
    

class ForwardEuler:
    
    def __init__(self):
        pass
    
    def step(self, rhsFunc, currentVal, t, stepSize):
        return currentVal + stepSize*rhsFunc(currentVal, t)


class BackwardEuler:
    
    forwardSolver = ForwardEuler()
    
    def __init__(self):
        pass
    
    def step(self, rhsFunc, currentVal, t, stepSize):
        
        """get an initial guess using a forward euler step"""
        xinit = self.forwardSolver.step(rhsFunc, currentVal, t, stepSize)
        tnext = t + stepSize
        
        """ backward euler step: find the optimal step by solving for the zero of the nonlinear system """
        objFunc = lambda(x): x - currentVal - stepSize*rhsFunc(x, tnext)
        nobjFunc = lambda(x): norm(objFunc(x))
           
        #nextVal = fsolve(objFunc, xinit)
        nextVal = fmin(nobjFunc, xinit, disp = False)
        #print 'nextVal={0}'.format(nextVal)        
        
        return nextVal

class RungeKutta4:
    
    def __init__(self):
        pass
    
    def step(self, rhsFunc, currentVal, t, stepSize):
        k1 = rhsFunc(currentVal, t)
        k2 = rhsFunc(currentVal + 0.5*stepSize*k1, t + 0.5*stepSize)
        k3 = rhsFunc(currentVal + 0.5*stepSize*k2, t + 0.5*stepSize)
        k4 = rhsFunc(currentVal + stepSize*k3, t + stepSize)
        
        nextVal = currentVal + (stepSize*(k1 + 2*k2 + 2*k3 + k4)) / 6;
        
        return nextVal

