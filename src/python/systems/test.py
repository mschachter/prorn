
import numpy as np

import matplotlib.pyplot as plt
from systems.core import LotkaVolterraSystem
from systems.neurons import IZNeuron, GenerateStepCurrent, HHNeuron, FitzHughNagumo, VanderPolOscillator
from systems.plots import Bounds, PhasePlot
from systems.simulation import Simulator

def testIZNeuronSimulator():
    
    iz = IZNeuron()
    iz.setCurrentWaveform(GenerateStepCurrent(100, 1000, 70))
        
    solvers = ['ForwardEuler', 'BackwardEuler', 'RungeKutta4']
        
    tstep = 1
    
    tsim = 250
    nsteps = int(round(tsim / tstep))
    print 'nsteps={0}'.format(nsteps)
    #print 'iz.initialState={0}'.format(repr(iz.initialState))
    #print 'sim.currentValue={0}'.format(repr(sim.currentValue))
            
    plt.figure()
    plt.hold(True)
    
    for s in solvers:
        
        sim = Simulator(iz, tstep)
        
        vals = np.zeros( (nsteps, 1) )
        vals[0] = sim.currentValue[0]    
        
        for n in range(nsteps):
            #sim.next()        
            sim.next(s)
            #print 'n={0}, value={1}'.format(n, repr(sim.currentValue))
            vals[n] = sim.currentValue[0]
        
        t = np.arange(0, tsim, tstep)
        plt.plot(t, vals)
        
    plt.legend(solvers, loc=2)
    plt.show()
    
    
def testHHNeuronSimulator(step_size=1e-6, sim_duration=0.030):
    
    tstep = step_size
    tsim = sim_duration
    nsteps = int(round(tsim / tstep))
    print 'nsteps={0}'.format(nsteps)
    
    hh = HHNeuron()
    hh.setCurrentWaveform(GenerateStepCurrent(0.050, tsim, 0e-9))
        
    #solvers = ['ForwardEuler', 'BackwardEuler', 'RungeKutta4']
    solvers = ['RungeKutta4']
    #solvers = ['ForwardEuler']
            
    #print 'hh.initialState={0}'.format(repr(hh.initialState))
            
    plt.figure()
    plt.hold(True)
    
    for s in solvers:
        
        sim = Simulator(hh, tstep)
        #print 'sim.currentValue={0}'.format(repr(sim.currentValue))
        
        vals = list()

        for n in range(nsteps):
            #sim.next()        
            sim.next(s)
            #print 'n={0}, value={1}'.format(n, repr(sim.currentValue))
            vals.append(sim.currentValue[0])
        
        t = np.arange(0.0, tsim, tstep)
        vals = np.array(vals)
        print 't.shape=',t.shape
        print 'vals.shape=',vals.shape
        plt.plot(t, vals)
        plt.axis([0, tsim, -0.080, 0.040])
                
    plt.legend(solvers, loc=2)
    plt.show()
    
    
def test_fitzhugh_nagumo(spacing=0.05, bounds=None, v0=0.0, w0=0.0, dc_current=0.0, solver='RungeKutta4'):

    fn = FitzHughNagumo(np.array([v0, w0]), dc_current)

    if bounds is None:
        bnds = Bounds()
        bnds.xmax = 1.0
        bnds.xmin = -1.0
        bnds.ymax = 1.0
        bnds.ymin = -0.5
    else:
        bnds = Bounds()
        bnds.xmin = bounds[0]
        bnds.xmax = bounds[1]
        bnds.ymin = bounds[2]
        bnds.ymax = bounds[3]

    pp = PhasePlot(fn, bnds, spacing)
    pp.renderPlot()

    timestep = 0.01
    sim = Simulator(fn, timestep)

    dur = 2000.00
    nsteps = int(dur / timestep)
    state = list()
    state.append(fn.initialState)
    for t in range(nsteps):
        x = sim.next(solverName=solver)
        state.append(x)

    state = np.array(state)
    print 'state.shape=',state.shape
    plt.figure()
    plt.plot(state[:, 0], 'k-')

    return state[:, 0]

def test_vanderpol(b=0.5, solver='RungeKutta4'):
    vdp = VanderPolOscillator(b=b)

    timestep = 1.0
    sim = Simulator(vdp, timestep)

    dur = 1000000.0
    nsteps = int(dur / timestep)
    state = list()
    state.append(vdp.initialState)
    for t in range(nsteps):
        x = sim.next(solverName=solver)
        state.append(x)

    state = np.array(state)
    print 'state.shape=',state.shape
    plt.figure()
    plt.plot(state[:, 0], 'k-')

    return state[:, 0]

