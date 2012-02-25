import re

import numpy as np

from sympy import Symbol, I, symbols, im

class SymbolicSchurSystem():
    
    def __init__(self, T):
        self.T = T
        self.N = T.shape[0]
    
    def construct(self):
        
        #construct symbolic unitary matrix U and it's inverse (conjugate transpose)
        U = np.zeros([self.N, self.N], dtype='object')
        Uinv = np.zeros([self.N, self.N], dtype='object')
        for i in range(self.N):
            for j in range(self.N):
                sindx = '%d%d' % (i, j)
                (a_ij, b_ij) = symbols('a_%s b_%s' % (sindx, sindx), complex=False)
                U[i, j] = a_ij + I*b_ij
                Uinv[j, i] = a_ij - I*b_ij
        self.U = np.matrix(U)
        self.Uinv = np.matrix(Uinv)
        
        #compute UU^{-1}
        self.UUinv = self.U * self.Uinv
        
        #compute UTU^{-1}
        self.UTUinv = self.U * self.T * self.Uinv
    
        #construct polynomial strings from the imaginary part of UTU^{-1}
        eqs = []
        p = re.compile(r're\(.*\)*')
        for i in range(self.N):
            for j in range(self.N):
                eq_str = str(im(self.UTUinv[i, j].expand()))
                istr = p.findall(eq_str)[0][3:-1]
                eqs.append(istr)
        
        #construct polynomials from UU^{-1} - I
        for i in range(self.N):
            for j in range(self.N):
                eq = self.UUinv[i, j]
                if i == j:
                    eq -= 1.0
                eq_str = str(eq.expand())
                eq_str = eq_str.replace('I', 'i')
                eqs.append(eq_str)
                
        self.polynomials = eqs
        