'''
Created on Oct 11, 2009

@author: mschachter
'''


from systems.core import System, ContinuousSystem

from numpy.core import array
import numpy

from math import pi


def GenerateStepCurrent(start, stop, val):
    ifunc = lambda(t): val if ((t >= start) and (t <= stop)) else 0        
    return ifunc


class Neuron(ContinuousSystem):
    
    def __init__(self):
        self.I = lambda (x): 0
    
    def setCurrentWaveform(self, ifunc):
        self.I = ifunc    
    
    def getOutput(self, outputNeuronId):
        pass



class IZNeuron(Neuron):
    
    def __init__(self, params = None):
        
        if (params == None):
            params = self.getDefaultParams()
                    
        self.vr = params['restVoltage']
        self.vt = params['thresholdVoltage']
        self.vpeak = params['peakVoltage']
        self.cap = params['capacitance']
        self.k = params['k']
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.d = params['d']        
        istate = array([self.vr, 0])

        Neuron.__init__(self)
        System.__init__(self, istate, self.rhs, params)


    def getPresynapticVoltage(self, state):
        return state[0]
        
                        
    def rhs(self, state, t):
        #print 'state={0}'.format(repr(state))
        v = state[0]
        u = state[1]
        iInj = self.I(t)
        #print 'iInj={0}'.format(iInj)
        dv = (self.k*(v - self.vr)*(v - self.vt) - u + iInj) / self.cap
        du = self.a*(self.b*(v - self.vr) - u)
        
        if (v+dv >= self.vpeak):
            dv = -v + self.c
            du += self.d
        
        return array([dv, du])       


    def getDefaultParams(self):
        
        params = {}
        
        params['restVoltage'] = -60 
        params['thresholdVoltage'] = -40        
        params['peakVoltage'] = 35
        params['capacitance'] = 100
        params['k'] = 0.7
        params['a'] = 0.03
        params['b'] = -2
        params['c'] = -50
        params['d'] = 100
        
        return params



class HHNeuron(Neuron):
    
    def __init__(self, params = None):
        
        if (params == None):
            params = self.getDefaultParams()
    
        istate = array([-0.060, 0, 0, 0])
    
        Neuron.__init__(self)
        System.__init__(self, istate, self.rhs, params)
    
    
    def rhs(self, state, t):
        
        v = state[0];
        m = state[1];
        h = state[2];
        n = state[3];
        
        cm = self.parameters['cm']
        gna = self.parameters['gna']
        gk = self.parameters['gk']
        gleak = self.parameters['gleak']
        narev = self.parameters['narev']
        krev = self.parameters['krev']
        leakrev = self.parameters['leakrev']
        ratemul = self.parameters['ratemul']
        condmul = self.parameters['condmul']
        
        Ina = condmul*gna*(m**3)*h*(narev - v);
        Ik = condmul*gk*(n**4)*(krev - v);
        Ileak = condmul*gleak*(leakrev - v);
        
        iInj = self.I(t)
        
        #print '\tIna={0}, Ik={1}, Ileak={2}, Iinj={3}'.format(Ina, Ik, Ileak, iInj)
        
        vmV = v*1e3;
        dv = ( Ina + Ik + Ileak + iInj ) / cm;
        dm = ratemul * 1e3 * (-(self.alpham(vmV) + self.betam(vmV))*m + self.alpham(vmV));
        dh = ratemul * 1e3 * (-(self.alphah(vmV) + self.betah(vmV))*h + self.alphah(vmV));
        dn = ratemul * 1e3 * (-(self.alphan(vmV) + self.betan(vmV))*n + self.alphan(vmV));
        
        return array([dv, dm, dh, dn])
    

    def alphah(self, v):
        return 0.4*numpy.exp( -(v+50)/20 )
    
    def alpham(self, v):         
        y = -.1*(v+30)

        if y is not 0:
            val = 6*y / (numpy.exp(y) - 1);
        else:
            #interpolate
            v1 = v - .001;
            y1 = -.1*(v1+30);
            val1 = 6*y1 / (numpy.exp(y1) - 1);
            
            v2 = v - .001;
            y2 = -.1*(v2+30);
            val2 = 6*y2 / (numpy.exp(y2) - 1);
            
            val = (val1 + val2) / 2;
        
        return val
    
    def alphan(self, v): 
        y = -.1*(v+40);
        val = .2*y / ( numpy.exp(y) - 1);
        
        return val
    
    def betah(self, v):
        return 6 / (numpy.exp( -.1*(v+20) ) + 1)

    def betam(self, v):
        return 20*numpy.exp( -(v +55)/18 )
    
    def betan(self, v):
        return .4*numpy.exp( -(v+50)/80 );
        
    def convertCapacitance(self, dia, cm):
        SA = pi*(dia**2) * 1e-6
        return cm * SA * 1e-6
    
    def convertDensity(self, dia, x):
        SA = pi*(dia**2) * 1e-6
        return x * SA
            
    def getDefaultParams(self):
        
        params = {}
        
        dia = 20

        spec_cm = 1
        cm = self.convertCapacitance(dia, spec_cm)
        
        nadens = 50e-3
        kdens = 12e-3
        leakdens = .01e-3
        
        gna = self.convertDensity(dia, nadens)
        gk = self.convertDensity(dia, kdens)
        gleak = self.convertDensity(dia, leakdens)
        
        params['cm'] = cm
        params['gna'] = gna
        params['gk'] = gk
        params['gleak'] = gleak 
        params['narev'] = 0.035
        params['krev'] = -0.075
        params['leakrev'] = -0.060
        params['ratemul'] = 1
        params['condmul'] = 1
        
        return params
