import csv

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

def sample_spacing_noise(proto_code):
    noisy_code = []
    for k,sym in enumerate(proto_code):
        noisy_code.extend(sym)
        if 

def sample_length_noise(proto_code):
    pass

def time_warp(code, speed_multiplier=2.0):
    pass

