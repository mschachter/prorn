import numpy as np
import matplotlib.pyplot as plt

def plot_prototypes(prototypes, noise_mean=0.0, noise_std=0.15):
    
    plen = len(prototypes)
    for k,sp in enumerate(prototypes):
        
        t = np.arange(0, sp.stim_len)
        nsamps = 5
        
        ax = plt.subplot(plen, 1, k+1)
        #plot prototype
        sproto = sp.get_prototype().squeeze()
        plt.plot(t, sproto, 'k-', linewidth=2)    
        #plot noisy instances    
        sfam = sp.generate_family(num_samps=nsamps, noise_mean=noise_mean, noise_std=noise_std)
        for k in range(nsamps):
            s = sfam[k, :].squeeze()
            plt.plot(t, s)
    
    plt.show()
  