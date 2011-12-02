import csv
import numpy as np

class NetworkInputStream():
    def __init__(self, shape):
        self.shape = shape        
    
    def next(self):
        return None
    
class CsvInputStream(NetworkInputStream):
    
    def __init__(self, shape, file_name, in_memory=True):        
        self.shape = shape
        self.row = 0
        self.in_memory = in_memory
        self.data = None
        self.file_name = file_name
        self.file_handle = open(file_name, 'r')
        self.csv_reader = csv.reader(self.file_handle)
        if self.in_memory:
            d = []
            rnum = 0
            for row in self.csv_reader:
                if len(row) > 0:                    
                    if len(row) != self.shape[1]:
                        print 'Row %d does not have right length: len(row)=%d, pre-specified shape=%s' % \
                                (self.row-1, len(row), str(self.shape))
                    frow = [float(r) for r in row]
                    d.append(frow)
                    rnum += 1            
            self.file_handle.close()
            self.data = np.array(d)
        
    
    def next(self):
        if self.in_memory:
            if self.row < self.data.shape[0]:
                self.row += 1
                return self.data[self.row-1, :].squeeze()
            return None
        else:
            r = self.csv_reader.next()
            if len(r) > 0:
                self.row += 1
                r = np.array(r)
                if r.shape != self.shape:
                    print 'Row %d does not have right shape, row shape=%s, pre-specified shape=%s' % \
                           (self.row-1, str(r.shape), str(self.shape))                 
                return r
            return None
                
    def close(self):
        if not in_memory:
            self.file_handle.close()
        else:
            del self.data
