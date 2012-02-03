import csv
import copy
import hashlib

import numpy as np

import h5py

class MorseLetter():
    def __init__(self):
        self.freq = None
        self.code = None
        self.symbol = None
    
    def get_samples(self, num_noisy_samples=20, num_jittered_samples=20, max_additions=1, max_jitter=1):
        noised_codes = {}
        num_noised_codes = min(num_noisy_samples, len(self.code)*(max_additions+1))
        if max_additions == 0:
            num_noised_codes = 1
        print 'code for %s: %s' % (self.symbol, str(self.code))
        print '# of possible noised codes: %d' % num_noised_codes
    
        #add binary noise to code themselves
        print '\tGenerated noised codes...'
        while len(noised_codes) < num_noised_codes:
            if max_additions > 0:      
                ncode = add_length_noise(self.code, max_additions=max_additions)
            else:
                ncode = self.code
            ncode_str = '_'.join([''.join([str(n) for n in note]) for note in ncode])
            if ncode_str not in noised_codes:
                noised_codes[ncode_str] = ncode
                
        #add jittering to noisy samples
        print '\tAdding temporal jitter...'
        jittered_samps = []
        for ncode_str,ncode in noised_codes.iteritems():
            jsamps = {}
            num_jittered_codes = 1
            if max_jitter > 0:
                num_jittered_codes = min(num_jittered_samples, (len(ncode)-1)*(max_jitter+1))
                
            if num_jittered_codes == 1:
                jcode = sample_with_jitter(ncode, min_spacing=2, max_deviation=0)
                jcode_str = ''.join([str(n) for n in jcode])
                jsamps[jcode_str] = jcode
            while len(jsamps) < num_jittered_codes:
                jcode = sample_with_jitter(ncode, min_spacing=2, max_deviation=max_jitter)
                jcode_str = ''.join([str(n) for n in jcode])
                if jcode_str not in jsamps:
                    jsamps[jcode_str] = jcode
            jittered_samps.extend(jsamps.values())
        
        return jittered_samps

class MorseStimSet:
    def __init__(self, stim_file=None, classes=['standard', 'time_warped', 'jitter', 'length_noise', 'jitter_and_length_noise']):
        self.classes = classes
        self.md5_to_symbol = {}
        self.symbol_to_md5 = {}
        self.md5_to_class = {}
        self.class_to_md5 = {}
        self.all_stims = {}
        if stim_file is not None:
            self.from_hdf5(stim_file)
    
    def from_hdf5(self, stim_file):
        fstim = h5py.File(stim_file, 'r')
    
        all_stims = {}
        for sc in self.classes:
            sc_grp = fstim[sc]
            self.class_to_md5[sc] = []
            for sym in sc_grp.keys():
                
                if sym not in self.symbol_to_md5:
                    self.symbol_to_md5[sym] = []
                sym_grp = sc_grp[sym]
                for stim_id in sym_grp.keys():       
                    stim_ds = sym_grp[stim_id]
                    md5 = stim_ds.attrs['md5']
                    self.symbol_to_md5[sym].append(md5)
                    self.md5_to_symbol[md5] = sym
                    self.md5_to_class[md5] = sc
                    self.class_to_md5[sc].append(md5)         
                             
                    samp = np.array(stim_ds, dtype='float')
                    if md5 in all_stims:
                        print 'WTF there are duplicate md5s: class=%s, stim_id=%s, md5=%s' % (sc, stim_id, md5)
                    all_stims[md5] = samp
        
        fstim.close()    
        self.all_stims = all_stims
    
def parse_from_file(file_name, dot_length=2, dash_length=4):

    letters = {}
    lens = {'d':dot_length, 's':dash_length}
    f = open(file_name, 'r')
    cr = csv.reader(f)
    for row in cr:
        if len(row) < 2:
            continue
        
        ltr = row[0]
        code = row[1]
        freq = float(row[2])
        
        bcode = []
        for k,c in enumerate(code):
            v = [1]*lens[c]
            bcode.append(v)
        ml = MorseLetter()
        ml.freq = freq
        ml.code = bcode
        ml.symbol = ltr
        letters[ltr] = ml
        
    f.close()
    
    return letters

def sample_with_jitter(proto_code, min_spacing=2, max_deviation=1):
    noisy_code = []
    for k,note in enumerate(proto_code):
        noisy_code.extend(note)
        if k < (len(proto_code)-1):
            snoise = np.random.randint(0, max_deviation+1)
            spacing = [0]*(min_spacing + snoise)
            noisy_code.extend(spacing)
    return noisy_code

def add_length_noise(proto_code, max_additions=1):
    ncode = []
    for k,note in enumerate(proto_code):
        nnote = copy.copy(note)
        snoise = np.random.randint(0, max_additions+1)
        if snoise > 0:
            nnote.extend([1]*snoise)
        ncode.append(nnote)
    return ncode



def write_stim_to_hdf5(code_file, output_file):
    
    f = h5py.File(output_file, 'w')
    
    #write frequencies
    letters_std = parse_from_file(code_file, dot_length=2, dash_length=6)
    freq_grp = f.create_group('frequencies')
    for sym,ltr in letters_std.iteritems():
        freq_grp[sym] = ltr.freq
    
    #write standard set of noiseless morse codes
    std_grp = f.create_group('standard')
    for sym,ltr in letters_std.iteritems():
        samp = sample_with_jitter(ltr.code, min_spacing=2, max_deviation=0)
        sym_grp = std_grp.create_group(sym)
        sym_grp['0'] = samp
        sym_grp['0'].attrs['md5'] = get_hash_for_sample(samp, 'standard')
    
    #write time-warped noiseless morse code
    tw_grp = f.create_group('time_warped')
    letters_tw = parse_from_file(code_file, dot_length=1, dash_length=3)
    for sym,ltr in letters_tw.iteritems():
        samp = sample_with_jitter(ltr.code, min_spacing=1, max_deviation=0)        
        sym_grp = tw_grp.create_group(sym)        
        sym_grp['0'] = samp
        sym_grp['0'].attrs['md5'] = get_hash_for_sample(samp, 'time_warped')

    #write morse code with length noise
    ln_grp = f.create_group('length_noise')
    for sym,ltr in letters_std.iteritems():
        ltr_grp = ln_grp.create_group(sym)
        samps = ltr.get_samples(num_noisy_samples=100, num_jittered_samples=100, max_additions=1, max_jitter=0)
        ltr_grp.attrs['count'] = len(samps)
        for k,samp in enumerate(samps):
            ltr_grp['%d' % k] = np.array(samp)
            ltr_grp['%d' % k].attrs['md5'] = get_hash_for_sample(samp, 'length_noise')
    
    #write morse code with temporal jitter
    tj_grp = f.create_group('jitter')
    for sym,ltr in letters_std.iteritems():
        ltr_grp = tj_grp.create_group(sym)
        samps = ltr.get_samples(num_noisy_samples=100, num_jittered_samples=100, max_additions=0, max_jitter=1)
        ltr_grp.attrs['count'] = len(samps)
        for k,samp in enumerate(samps):            
            ltr_grp['%d' % k] = np.array(samp)
            ltr_grp['%d' % k].attrs['md5'] = get_hash_for_sample(samp, 'jitter')
            
    #write morse code with both length noise and temporal jitter
    lntj_grp = f.create_group('jitter_and_length_noise')
    for sym,ltr in letters_std.iteritems():
        ltr_grp = lntj_grp.create_group(sym)
        samps = ltr.get_samples(num_noisy_samples=100, num_jittered_samples=100, max_additions=1, max_jitter=1)
        ltr_grp.attrs['count'] = len(samps)
        for k,samp in enumerate(samps):
            ltr_grp['%d' % k] = np.array(samp)
            ltr_grp['%d' % k].attrs['md5'] = get_hash_for_sample(samp, 'jitter_and_length_noise')
    
    f.close()
    


def get_hash_for_sample(samp, family=None):
    hash = hashlib.md5()
    if family is not None:
        hash.update(family)
    hash.update(''.join([str(x) for x in samp]))
    return hash.hexdigest()
    