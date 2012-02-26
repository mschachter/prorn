import re
import copy

import numpy as np

from sympy import Symbol, I, symbols, im
from sympy.parsing.sympy_parser import parse_expr

class SymbolicSchurSystem():
    
    def __init__(self, T):
        self.T = T
        self.N = T.shape[0]
        self.construct()
    
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
                eqs.append(eq_str)
                
        self.polynomials = eqs
    
    def create_symbolic_T(self):
        T = np.zeros([self.N, self.N], dtype='object')
        for i in range(self.N):
            for j in range(self.N):
                sindx = '%d%d' % (i, j)
                (a_ij, b_ij) = symbols('ta_%s tb_%s' % (sindx, sindx), complex=False)
                T[i, j] = a_ij + I*b_ij
        return T
        

    def test(self, U):
        """ Test the system of equations for a given U """
        uvals = {}
        for i in range(self.N):
            for j in range(self.N):
                sindx = '%d%d' % (i, j)
                a = U[i, j].real
                b = U[i, j].imag
                uvals['a_%s' % sindx] = '(%0.16f)' % a 
                uvals['b_%s' % sindx] = '(%0.16f)' % b
        
        for eq in self.polynomials:
            eqnum = copy.copy(eq)
            for vname,vval in uvals.iteritems():
                eqnum = eqnum.replace(vname, vval)
            eqval = parse_expr(eqnum)
            
            print '---------------'
            print 'Expression: %s' % eq
            #print 'Filled in: %s' % eqnum
            print 'Value: %s' % str(eqval)

    def to_bertini(self, output_file):
        f = open(output_file, 'w')
        f.write('CONFIG\n')
        f.write('END;\n')
        f.write('INPUT\n')
        f.write('variable_group ')
        vgrps = []
        for i in range(self.N):
            for j in range(self.N):
                sindx = '%d%d' % (i, j)
                vgrps.append('a_%s' % sindx)
                vgrps.append('b_%s' % sindx)
        f.write('%s;\n' % ','.join(vgrps))
        f.write('function ')
        nfuncs = 2*self.N**2
        fstrs = ['f%d' % k for k in range(nfuncs)]
        f.write('%s;\n' % ','.join(fstrs))
        
        for k,eq in self.polynomials:
            eq.replace('**', '^')
            f.write('%s=%s;\n' % (fstrs[k], eq))
        f.write('END;')
        f.close()
        