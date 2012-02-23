

class SymbolicMatrix():
    def __init__(self):
        pass
    


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
            
            new_mat = []
            
            for i in range(self.current_matrix.shape[0]):
                for j in range(M.shape[1]):
                    row = self.current_matrix[i, :]
                    col = M[:, j]
                    new_str = []
                    for k in M.shape[0]:
                        pass
    