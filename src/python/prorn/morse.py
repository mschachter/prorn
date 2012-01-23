import csv

import numpy as np


def encode_from_file(file_name, dot_length=2, dash_length=4, spacing=2):

    codes = {}
    lens = {'d':dot_length, 's':dash_length}
    f = open(file_name, 'r')
    cr = csv.reader(f)
    for row in cr:
        if len(row) < 2:
            continue
        
        ltr = row[0]
        code = row[1]
        
        bcode = []
        for k,c in enumerate(code):
            l = lens[c]
            v = [1]*l
            bcode.extend(v)
            if k != len(code)-1:
                s = [0]*spacing
                bcode.extend(s)
        codes[ltr] = bcode
        
    f.close()
    
    return codes
