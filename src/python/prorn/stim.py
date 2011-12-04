import numpy as np
import h5py

class StimPrototype:
    
    def __init__(self, stim_len=15):
        self.stim_len = stim_len
        self.bump_times = []
        self.bump_heights = []
        self.bump_widths = []
    
    def get_prototype(self):
        t = np.arange(0, self.stim_len)
        stim_base = np.zeros([1, self.stim_len])
                
        for k,tbump in enumerate(self.bump_times):
            bamp = self.bump_heights[k]
            bstd = self.bump_widths[k]        
            b = bamp * np.exp(-(t-tbump)**2 / (2*np.pi*bstd))
            stim_base += b
            
        return stim_base
    
    def to_hdf5(self, f, key):
        """ Write this object to an hdf5 file """
        
        grp = f.create_group(key)
        grp['bump_times'] = self.bump_times
        grp['bump_heights'] = self.bump_heights
        grp['bump_widths'] = self.bump_widths
        grp['stim_len'] = self.stim_len
        grp.attrs['type'] = '%s.%s' % (self.__module__, self.__class__.__name__)
        
    def from_hdf5(self, f, key):
        
        grp = f[key]
        tname = '%s.%s' % (self.__module__, self.__class__.__name__)
        if tname != grp.attrs['type']:
            print 'Can\'t read object from hdf5 group %s, type name is wrong. %s != %s' % \
                    (key,  grp.attrs['type'], tname)
            return
         
        self.bump_times = np.array(grp['bump_times'])
        self.bump_heights = np.array(grp['bump_heights'])
        self.bump_widths = np.array(grp['bump_widths'])
        self.stim_len = np.array(grp['stim_len'])
            
        
    def generate_family(self, num_samps=10, noise_mean=0.0, noise_std=0.15):
        """ Generate a family of 1D time series.
        
            stim_len: length in time steps of base stimulus
            bump_times: a gaussian bump is centered at each time point in this array
            bump_heights: each gaussian is scaled by a corresponding magnitude in this array
            bump_widths: each gaussian has a width set by an element in this array
            num_samps: the number of samples to generate
            noise_mean: mean gaussian noise added to each sample
            noise_std: std of gaussian noise added to each sample    
        """
        
        stim_base = self.get_prototype()
        
        stims = []
        for k in range(num_samps):
            stim = np.copy(stim_base)
            noise = np.random.randn(1, self.stim_len)        
            stim += noise*noise_std + noise_mean
            stim[stim < 0.0] = 0.0        
            stims.append(stim.squeeze()) 
        
        return np.array(stims)


def generate_prototype(num_bumps, stim_len, padding=1, max_height=1, num_heights=3, mean_width=0.1):
    
    bump_times = []
    bump_heights = []
    bump_widths = []
    while len(bump_times) < num_bumps:
        tbump = np.random.randint(stim_len-2*padding) + padding
        if tbump not in bump_times:
            #generate height and width
            bheight = np.random.randint(num_heights+1)
            bheight /= max_height            
            bstd = mean_width*np.abs(np.random.randn())
            bump_heights.append(bheight)
            bump_widths.append(bstd)
            bump_times.append(tbump)

    sp = StimPrototype(stim_len)
    sp.bump_times = bump_times
    sp.bump_heights = bump_heights
    sp.bump_widths = bump_widths
    
    return sp


def generate_prototypes(num_stims, stim_len=10, max_bump_frac=0.50, num_heights=3, max_height=1, mean_width=0.1):
    
    bump_time_combos = {}
    prototypes = []
    while len(prototypes) < num_stims:        
        num_bumps = int(np.ceil(max_bump_frac*np.random.randint(stim_len-1)) + 1)
        #randomly generate bumps
        sp = generate_prototype(num_bumps, stim_len, max_height=max_height, num_heights=num_heights, mean_width=mean_width)        
        btc_key = tuple(sp.bump_times)
        if btc_key not in bump_time_combos:
            bump_time_combos[btc_key] = True
            prototypes.append(sp)
    
    return prototypes

def write_stims_to_file(prototypes, output_file, num_samps=50, noise_mean=0.0, noise_std=0.15):
    
    f = h5py.File(output_file, 'w')
    
    for k,sp in enumerate(prototypes):
        key = 'stim_%d' % k
        sp.to_hdf5(f, key)
        sfam = sp.generate_family(num_samps=num_samps, noise_mean=noise_mean, noise_std=noise_std)
        f[key]['samples'] = sfam
        f[key]['samples'].attrs['num_samps'] = num_samps
        f[key]['samples'].attrs['noise_mean'] = noise_mean
        f[key]['samples'].attrs['noise_std'] = noise_std
    
    f.close()

def read_stims_from_file(sample_file):
    
    f = h5py.File(sample_file, 'r')
    
    stims = {}
    for stim_key in f.keys():
        stims[stim_key] = np.array(f[stim_key]['samples'])
    
    f.close()
    
    return stims
    
    
    
    