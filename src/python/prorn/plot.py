import os
import time
import operator

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg

import h5py

from prorn.stim import StimPrototype
from prorn.readout import get_samples
from prorn.analysis import get_perfs



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
  
def plot_stimuli(stim_file):
    
    f = h5py.File(stim_file, 'r')
    
    stim_keys = f.keys() 
    plen = len(stim_keys)
    
    for k,stim_key in enumerate(stim_keys):
        
        ax = plt.subplot(plen, 1, k+1)
        
        sp = StimPrototype()
        sp.from_hdf5(f, stim_key)
        
        stim_samps = np.array(f[stim_key]['samples'])
        nsamps = stim_samps.shape[0]        
                
        sproto = sp.get_prototype().squeeze()        
        t = np.arange(0, sp.stim_len)
        plt.plot(t, sproto, 'k-', linewidth=2)
        
        for k in range(nsamps):
            s = stim_samps[k, :].squeeze()
            plt.plot(t, s)
            
    f.close()
    plt.show()

  
def plot_single_net_and_stim_movie(stim_file, net_file, net_key, stim_key, trial, temp_dir, output_file):
    
    #get stim
    fstim = h5py.File(stim_file, 'r')
    stim_samps = np.array(fstim[stim_key]['samples'])
    fstim.close()
    
    fnet = h5py.File(net_file, 'r')
    net_stims = np.array(fnet[net_key][stim_key])
    stim_start = fnet[net_key][stim_key].attrs['stim_start']
    stim_end = fnet[net_key][stim_key].attrs['stim_end']
    fnet.close()
    
    net_state = net_stims[trial, :, :]
    total_len = net_state.shape[0]
    
    stim = np.zeros(total_len)     
    stim[stim_start:stim_end] = stim_samps[trial, :]
        
    fig = plt.figure(figsize=(10, 5))
    
    ax_stim = fig.add_subplot(1, 2, 1)
    ax_state = fig.add_subplot(1, 2, 2, projection='3d')
    
    t_rng = np.arange(total_len)
    ax_stim.set_xlim([0, t_rng.max()])
    ax_stim.set_ylim([0, stim.max()])
    
    x1 = net_state[:, 0].squeeze()
    x2 = net_state[:, 1].squeeze()
    x3 = net_state[:, 2].squeeze()
    
    initial_state = net_state[stim_start-1]
    
    #ax_state.set_xlim3d([x2.min(), x2.max()])
    #ax_state.set_ylim3d([x3.min(), x3.max()])
    #ax_state.set_zlim3d([x1.min(), x1.max()])
    ax_state.set_xlabel('X_2')
    ax_state.set_ylabel('X_3')
    ax_state.set_zlabel('X_1')
    
    ax_state.scatter([x2.min(), x2.max()], [x3.min(), x3.max()], [x1.min(), x1.max()], c='w', s=1)
    
    fig_prefix = '%s_%s_%d_%%d' % (net_key, stim_key, trial)
    fig_prefix_rm = '%s_%s_%d_*' % (net_key, stim_key, trial)
    fig_path = os.path.join(temp_dir, '%s.png' % fig_prefix)
    
    for t in t_rng:
        
        state_dist = np.linalg.norm(net_state[t, :] - initial_state)
        
        ax_stim.plot(t_rng[0:t+1], stim[0:t+1], 'k-', linewidth=2)
        clr = 'b'
        sz = 16
        if t == 0:
            clr = 'r'
            sz = 36            
        ax_state.scatter([x2[t]], [x3[t]], [x1[t]], c=clr, s=sz)
        ax_stim.set_title('t=%d, dist=%0.3f' % (t, state_dist))
        save_to_png(fig, fig_path % t)

    mov_cmd = 'ffmpeg -r 4 -i %s %s' % (fig_path, output_file)
    print 'Running %s' % mov_cmd
    os.system(mov_cmd)
    
    rm_cmd = 'rm %s' % os.path.join(temp_dir, fig_prefix_rm)
    print 'Removing temp files...'
    os.system(rm_cmd)
    

def create_net_and_stim_movies(stim_file, net_file, temp_dir='/home/cheese63/temp', output_dir='/home/cheese63/net_movies', limit=None):
    
    ntrials = 20
    fstim = h5py.File(stim_file, 'r')
    stim_keys = fstim.keys()
    fstim.close()
    
    fnet = h5py.File(net_file, 'r')
    net_keys = fnet.keys()
    fnet.close()
    
    if limit is not None:
        net_keys = net_keys[0:limit]
    
    num_movies = len(stim_keys)*len(net_keys)*ntrials
    num_movies_left = num_movies
    
    for net_key in net_keys:
        dirname = os.path.join(output_dir, net_key)
        try:
            os.mkdir(dirname)
        except:
            pass
        for stim_key in stim_keys:
            for trial in range(ntrials):
                
                stime = time.time()                
                output_file = os.path.join(dirname, '%s_%s_%d.mp4' % (net_key, stim_key, trial))
                plot_single_net_and_stim_movie(stim_file, net_file, net_key, stim_key, trial, temp_dir, output_file)
                
                etime = time.time() - stime
                print '-----------------------'
                print 'Movie %d of %d, %0.1fs, %0.1f min left' % \
                      (num_movies-num_movies_left, num_movies, etime, float(num_movies_left-1)*etime / 60.0)
                print '-----------------------'
                num_movies_left -= 1


def plot_readout_data(net_file, net_key, readout_window):

    clrs = ['r', 'g', 'b', 'k', 'c']
    
    (samps, all_stim_classes, Ninternal) = get_samples(net_file, net_key, readout_window)
    
    states = np.array([x[0] for x in samps])
    stim_classes = np.array([x[1] for x in samps])
        
    fig = plt.figure()    
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    for k,sc in enumerate(all_stim_classes):
        sc_states = states[stim_classes == sc, :]    
        ax.scatter(sc_states[:, 0], sc_states[:, 1], sc_states[:, 2], c=clrs[k])
    
    plt.show()
    

def plot_perf_histograms(net_files):
    
    perfs = get_perfs(net_files)    
    indx = np.isnan(perfs[:, 0]) | np.isnan(perfs[:, 1]) | np.isnan(perfs[:, 2])
    
    plt.clf()
    fig = plt.gcf()
    
    ax = fig.add_subplot(3, 1, 1)
    ax.hist(perfs[indx == False, 0], bins=25)
    ax.set_title('NN Performance')
    
    ax = fig.add_subplot(3, 1, 2)
    ax.hist(perfs[indx == False, 1], bins=25)
    ax.set_title('Logit Performance')
    
    ax = fig.add_subplot(3, 1, 3)
    ax.hist(perfs[indx == False, 2], bins=25)
    ax.set_title('Entropy Ratio')
    
    plt.show()
    

def plot_entropy_ratio_vs_perf(net_files):

    (perfs, index2keys) = get_perfs(net_files)    
    indx = np.isnan(perfs[:, 0]) | np.isnan(perfs[:, 1]) | np.isnan(perfs[:, 2])
    nn_perf = perfs[indx == False, 0].squeeze()
    logit_perf = perfs[indx == False, 1].squeeze()
    entropy_ratio = perfs[indx == False, 2].squeeze()
    
    nn_list = zip(nn_perf, index2keys)
    nn_list.sort(key=operator.itemgetter(0), reverse=True)
    
    logit_list = zip(logit_perf, index2keys)
    logit_list.sort(key=operator.itemgetter(0), reverse=True)
    
    er_list = zip(entropy_ratio, index2keys)
    er_list.sort(key=operator.itemgetter(0))
    
    print '--------- Top 50 ---------'
    print 'NN\t\t\tLogit\t\t\tEntropy Ratio'
    for k in range(50):
        print '%d) %0.2f,%s\t%0.0f,%s\t%0.2f,%s' % ((k+1), nn_list[k][0], nn_list[k][1], \
                                                    logit_list[k][0], logit_list[k][1], \
                                                    er_list[k][0], er_list[k][1])
    
        
    plt.clf()
    fig = plt.gcf()

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(logit_perf, nn_perf, 'ko')
    plt.xlabel('Logit Perf')
    plt.ylabel('NN Perf')
    plt.axis('tight')

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(entropy_ratio, nn_perf, 'go')
    plt.xlabel('Entropy Ratio')
    plt.ylabel('NN Perf')
    plt.axis('tight')
    
    ax = fig.add_subplot(3, 1, 3)
    ax.plot(entropy_ratio, logit_perf, 'bo')
    plt.xlabel('Entropy Ratio')
    plt.ylabel('Logit Perf')
    plt.axis('tight')
    
    plt.show()
    

def save_to_png(fig, output_file):
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(output_file, dpi=72)
    