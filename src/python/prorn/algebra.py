import numpy as np

class SystemOfEquations():
    def __init__(self):
        self.current_matrix = None
        
    def multiply_matrix(self, M):
        if self.current_matrix is None:
            self.current_matrix = M
        else:
            if self.current_matrix.shape[1] != M.shape[0]:
                print '# of cols != # of rows (%d != %d)' % (self.current_matrix.shape[1], M.shape[0])
                return
            
            new_mat = np.zeros([self.current_matrix.shape[0], M.shape[1]], dtype='string')
            
            for i in range(self.current_matrix.shape[0]):
                for j in range(M.shape[1]):
                    lrow = self.current_matrix[i, :]
                    rcol = M[:, j]
                    
                    entry_str = []
                    
                    for k,lterm in enumerate(lrow):
                        #split left hand entry into additive parts
                        aparts = lterm.split('+')
                        #distribute right hand term with left hand terms
                        strs = []
                        rterm = rcol[k]
                        for a in aparts:
                            strs.append('%s*%s' % (rterm, a))
                        entry_str.append('+'.join(strs))
                    
                    #join all the terms together for this entry
                    new_mat[i, j] = '+'.join(entry_str)
            
            self.current_matrix = new_mat