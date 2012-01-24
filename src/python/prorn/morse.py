import csv
import copy
import numpy as np


def parse_from_file(file_name, dot_length=2, dash_length=4):

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
            v = [1]*lens[c]
            bcode.append(v)
        codes[ltr] = bcode
        
    f.close()
    
    return codes

def sample_with_jitter(proto_code, mean_spacing=2, max_deviation=1):
    noisy_code = []
    for k,sym in enumerate(proto_code):
        noisy_code.extend(sym)
        if k < len(proto_code):
            snoise = np.random.randint(0, max_deviation+1)
            spacing = [0]*(mean_spacing + snoise)
            noisy_code.extend(spacing)
    return noisy_code

def add_length_noise(proto_code, max_additions=1):
    ncode = []
    for k,sym in enumerate(proto_code):
        nsym = copy.copy(sym)
        snoise = np.random.randint(0, max_additions+1)
        if snoise > 0:
            nsym.extend([1]*snoise)
        ncode.append(nsym)
    return ncode
