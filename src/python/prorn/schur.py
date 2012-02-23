import numpy as np

import scipy
from scipy.optimize import fsolve,broyden1,broyden2,newton_krylov,anderson,fmin_cg,fmin_bfgs
import operator

class SchurEstimator:
    """ Estimates a real-valued weight matrix from the upper triangular part of a Schur decomposition """
    
    def __init__(self, T):
        self.T = np.matrix(T)
        self.N = T.shape[0]
        self.shape = T.shape
    
    def objective(self, x):
        
        U = np.matrix(x.reshape(self.shape), dtype='complex')
        
        UT = U.conj().transpose()
        top = np.imag(U*self.T*UT)
        bottom = U*UT - np.eye(self.N)
        
        eq_all = top + bottom
        eq = np.array(eq_all).ravel()
        
        return eq
    
    def objective_lsq(self, x):
        return np.linalg.norm(self.objective(x))
        
    
    def estimate(self, x0=None):
        if x0 is None:
            nvals = self.shape[0]*self.shape[1]
            x0 = np.zeros([nvals], dtype='complex')
            for k in range(nvals):
                x0[k] = np.complex(np.random.randn()*1e-2, np.random.randn()*1e-2)
        
        soln = newton_krylov(self.objective, x0, verbose=True)
        self.final_error = self.objective_lsq(soln)
        self.U = soln.reshape(self.shape)
    
    def estimate_lsq(self, x0=None):
        if x0 is None:
            nvals = self.shape[0]*self.shape[1]
            x0 = np.zeros([nvals], dtype='complex')
            for k in range(nvals):
                x0[k] = np.complex(np.random.randn()*1, np.random.randn()*1)
        
        #(xopt, fopt, func_calls, grad_calls, warn_flag) = fmin_cg(self.objective_lsq, x0, full_output=True, gtol=1e-12)
        xopt = fmin_bfgs(self.objective_lsq, x0)
        self.final_error = self.objective_lsq(xopt)
        self.U = xopt.reshape(self.shape)
    
    
def test_schur_estimator():
    
    W = np.array([[0.55846016, 0.61462797],
                  [-0.999,  0.66074835]])
    
    (T, U) = scipy.linalg.schur(W, 'complex')
    
    schur_est = SchurEstimator(T)
    
    num_lsq_ests = 1
    num_root_ests = 500
    
    total_ests = []
    
    #x0 = np.array(U).ravel().squeeze()
    for k in range(num_lsq_ests):
        schur_est.estimate_lsq(None)
        total_ests.append((schur_est.final_error, schur_est.U))
    for k in range(num_root_ests):
        try:
            schur_est.estimate(None)
            total_ests.append((schur_est.final_error, schur_est.U))
        except:
            pass
    
    total_ests.sort(key=operator.itemgetter(0))
    print total_ests
    
    top_ests = np.array([est[1] for est in total_ests[:10]])
    #Uest = np.matrix(top_ests.mean(axis=0))
    Uest = np.matrix(top_ests[0].squeeze())
    
    print 'Actual U:'
    print U
    print 'Estimated U:'
    print Uest

    West = np.real(Uest * T * Uest.conj().transpose())
    print 'Actual W:'
    print W
    print 'Estimated W:'
    print West
