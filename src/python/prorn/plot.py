import os
import time
import operator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg

import h5py

from prorn.stim import StimPrototype, stim_pca
from prorn.readout import get_samples
from prorn.spectra import plot_pseudospectra
from prorn.analysis import get_perfs, filter_perfs, top100_weight_pca, get_info_data


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

  
def plot_info_data(net_files):
    
    idata = get_info_data(net_files)
    
    plt.clf()
    fig = plt.gcf()
    
    for k,(nbins,ilist) in enumerate(idata.iteritems()):
        H = np.array([x[0] for x in ilist])
        MI = np.array([x[1] for x in ilist])
        
        nzi = MI >= 0.0
        print '# of positive MIs: %d' % len(nzi.nonzero()[0])
        
        sp_indx = k*2 + 1
        
        ax = fig.add_subplot(len(idata), 2, sp_indx)
        ax.hist(H[nzi], bins=20)
        ax.set_xlabel('Entropy (nbins=%d)' % nbins)
        plt.axis('tight')
        
        ax = fig.add_subplot(len(idata), 2, sp_indx+1)
        ax.hist(MI[nzi], bins=20)
        ax.set_xlabel('Mutual Information (nbins=%d)' % nbins)
        plt.axis('tight')
            
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
    ax.set_title('Post-Stimulus Network State')
    plt.axis('tight')
    
    plt.show()
    

def plot_perf_histograms(pdata, filter=True):
    
    if filter:
        pdata = filter_perfs(pdata)
    
    nn_perfs = np.array([p.nn_perf for p in pdata])
    logit_perfs = np.array([p.logit_perf for p in pdata])
    ers = np.array([p.entropy_ratio for p in pdata])
    
    plt.clf()
    
    fig = plt.gcf()    
    ax = fig.add_subplot(3, 1, 1)
    ax.hist(nn_perfs, bins=25)
    ax.set_title('% Correct (NN Readout)')
    plt.axis('tight')
    
    ax = fig.add_subplot(3, 1, 2)
    ax.hist(logit_perfs, bins=25)
    ax.set_title('% Correct (Logit Readout)')
    plt.axis('tight')
    
    ax = fig.add_subplot(3, 1, 3)
    ax.hist(ers, bins=25)
    ax.set_title('Entropy Ratio')
    plt.axis('tight')
    
    plt.show()
    
def plot_mi_vs_eig_vs_perf(pdata, filter=True):
    if filter:
        pdata = filter_perfs(pdata, mi_cutoff=0.0)
        
    nn_perfs = np.array([p.nn_perf for p in pdata])    
    mi = np.array([p.mutual_information for p in pdata])
    
    nn_perfs_norm = 1.0 - (nn_perfs / nn_perfs.max())
    
    ev1 = []
    ev2 = []
    for p in pdata:
        ev1.append([p.eigen_values[0].real, p.eigen_values[0].imag])
        ev2.append([p.eigen_values[1].real, p.eigen_values[1].imag])
    ev1 = np.array(ev1)
    ev2 = np.array(ev2)   
    ev1_mod = np.sqrt((ev1**2).sum(axis=1))    
    ev2_mod = np.sqrt((ev2**2).sum(axis=1))

    plt.clf()
    fig = plt.gcf()
    
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(ev1_mod, ev2_mod, mi)
    ax.set_xlabel('$|\lambda_1|$')
    ax.set_ylabel('$|\lambda_2|$')
    ax.set_zlabel('Mutual Information')
    
    """
    plt.clf()
    fig = plt.gcf()
    
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for k in range(len(nn_perfs_norm)):
        r = nn_perfs_norm[k]
        g = 0
        b = 1.0 - nn_perfs_norm[k]
        ax.scatter([ev1_mod[k]], [ev2_mod[k]], [mi[k]], c=(r, g, b), s=30)
    ax.set_xlabel('$|\lambda_1|$')
    ax.set_ylabel('$|\lambda_2|$')
    ax.set_zlabel('Mutual Information')
    """
    
    plt.show()
    
    

def plot_entropy_ratio_vs_perf(pdata, filter=True):
    
    if filter:
        pdata = filter_perfs(pdata, mi_cutoff=0.0)

    nn_perfs = 1.0 - (np.array([p.nn_perf for p in pdata]) / 100.0)
    logit_perfs = 1.0 - np.array([p.logit_perf for p in pdata])
    mi = np.array([p.mutual_information for p in pdata])
    entropy = np.array([p.entropy for p in pdata])
    entropy_ratios = np.array([p.entropy_ratio for p in pdata])
                
    plt.clf()
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(logit_perfs, nn_perfs, 'o', markerfacecolor='gray')
    plt.xlabel('% Correct (Logit Readout)')
    plt.ylabel('% Correct (NN Readout)')
    plt.axis('tight')
    
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(mi, nn_perfs, 'o', markerfacecolor='orange')
    plt.xlabel('Mutual Information (bits)')
    plt.ylabel('% Correct (NN Readout)')
    plt.axis('tight')
    
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(mi, logit_perfs, 'o', markerfacecolor='orange')
    plt.xlabel('Mutual Information (bits)')
    plt.ylabel('% Correct (Logit Readout)')
    plt.axis('tight')
    
    plt.show()

def plot_top100_weights_cov(pdata):

    (weights, wcov, evals, evecs, proj, nn_perfs) = top100_weight_pca(pdata)
    
    plt.clf()
    fig = plt.gcf()
    for k in range(9):
        ax = fig.add_subplot(3, 3, k+1)
        weig = evecs[k].reshape([3, 3])
        wmax = np.abs(weig).max()
        ax.imshow(weig, interpolation='nearest', cmap=cm.jet, vmin=-wmax, vmax=wmax)
        ax.set_title('%0.4f' % evals[k])
    
    wcov_nodiag = wcov
    for k in range(9):
        wcov_nodiag[k, k] = 0.0
    fig = plt.figure()
    wcovmax = np.abs(wcov_nodiag).max()
    ax = fig.add_subplot(1, 1, 1)
    res = ax.imshow(wcov_nodiag, interpolation='nearest', cmap=cm.jet, vmin=-wcovmax, vmax=wcovmax)
    fig.colorbar(res)
    ax.set_title('Weight Covariance (diagonal removed)')
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(proj[:, 0], proj[:, 1], nn_perfs)
    ax.set_title('PCA Weight Projection vs NN Perf')
        
    plt.show()
     

def plot_top_100_weights(pdata, filter=True, rootdir='/home/cheese63/git/prorn/data'):
    if filter:
        pdata = filter_perfs(pdata)
        
    pdata.sort(key=operator.attrgetter('nn_perf'))
    weights = []
    for p in pdata[0:100]:
        fname = os.path.join(rootdir, p.file_name)
        net_key = p.net_key
        f = h5py.File(fname, 'r')
        W = np.array(f[net_key]['W'])
        weights.append(W)        
        f.close()
    
    weights = np.array(weights)
    
    wmean = weights.mean(axis=0).squeeze()
    wstd = weights.std(axis=0).squeeze()
    
    plt.clf()
    fig = plt.gcf()
    
    perrow = 10
    percol = 10
    
    fig = plt.gcf()    
    for k in range(100):
        #r = np.floor(k / perrow) + 1
        #c = (k % percol) + 1
        ax = fig.add_subplot(perrow, percol, k)
        ax.imshow(weights[k, :, :], interpolation='nearest', vmin=-1.0, vmax=1.0, cmap=cm.jet)
        plt.xticks([], [])
        plt.yticks([], [])
        
        
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    res = ax.imshow(wmean, interpolation='nearest', vmin=-1.0, vmax=1.0, cmap=cm.jet)
    ax.set_title('Mean weights for top 100')
    fig.colorbar(res)
    
    ax = fig.add_subplot(1, 2, 2)
    res = ax.imshow(wstd, interpolation='nearest', cmap=cm.jet)
    fig.colorbar(res)
    ax.set_title('Std of weights for top 100')
    
    fig = plt.figure()
    for k in range(9):
        r = np.floor(k / 3)
        c = k % 3
        wvals = weights[:, r, c].squeeze()
        
        ax = fig.add_subplot(3, 3, k+1)
        ax.hist(wvals, bins=15)
        ax.set_title('Weight (%d, %d)' % (r, c))
        plt.axis('tight')
    
    plt.show()


def plot_eigenvalues_vs_perf(pdata, filter=True):
    
    if filter:
        pdata = filter_perfs(pdata)
        
    nn_perfs = np.array([p.nn_perf for p in pdata])
    logit_perfs = np.array([p.logit_perf for p in pdata])
        
    ev1 = []
    ev2 = []
    ev3 = []
    for p in pdata:
        ev1.append([p.eigen_values[0].real, p.eigen_values[0].imag])
        ev2.append([p.eigen_values[1].real, p.eigen_values[1].imag])
        ev3.append([p.eigen_values[2].real, p.eigen_values[2].imag])
    ev1 = np.array(ev1)
    ev2 = np.array(ev2)
    ev3 = np.array(ev3)
    
    ev1_mod = np.sqrt((ev1**2).sum(axis=1))
    ev2_mod = np.sqrt((ev2**2).sum(axis=1))
    ev3_mod = np.sqrt((ev3**2).sum(axis=1))
    
    plt.clf()
    
    maxev = max([ev1_mod.max(), ev2_mod.max(), ev3_mod.max()])
    minev = min([ev1_mod.min(), ev2_mod.min(), ev3_mod.min()])
    fig = plt.gcf()
    ax = fig.add_subplot(3, 1, 1)
    ax.hist(ev1[:, 0], facecolor='orange', bins=15)
    plt.xlim([-minev, maxev])
    ax.set_title('$|\lambda_1|$', fontsize=24)
    ax = fig.add_subplot(3, 1, 2)
    ax.hist(ev2[:, 0], facecolor='orange', bins=15)
    plt.xlim([-minev, maxev])
    ax.set_title('$|\lambda_2|$', fontsize=24)
    ax = fig.add_subplot(3, 1, 3)
    ax.hist(ev3[:, 0], facecolor='orange', bins=15)
    plt.xlim([-minev, maxev])
    ax.set_title('$|\lambda_3|$', fontsize=24)    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(ev1_mod, nn_perfs, 'ro')
    ax.set_title('Eigenvalue 1')
    plt.xlabel('|e1|')
    plt.ylabel('NN Perf')
    plt.axis('tight')
    
    ax = fig.add_subplot(3, 1, 2)
    ax.plot(ev2_mod, nn_perfs, 'ro')
    ax.set_title('Eigenvalue 2')
    plt.xlabel('|e2|')
    plt.ylabel('NN Perf')
    plt.axis('tight')
    
    ax = fig.add_subplot(3, 1, 3)
    ax.plot(ev3_mod, nn_perfs, 'ro')
    ax.set_title('Eigenvalue 3')
    plt.xlabel('|e3|')
    plt.ylabel('NN Perf')
    plt.axis('tight')
    
    
    fig = plt.figure()        
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(ev1[:, 0], ev1[:, 1], nn_perfs, 'ko')
    ax.set_title('Eigenvalue 1')
    plt.xlabel('Real')
    plt.ylabel('Imag')
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(ev2[:, 0], ev2[:, 1], nn_perfs, 'ko')
    ax.set_title('Eigenvalue 2')
    plt.xlabel('Real')
    plt.ylabel('Imag')
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(ev3[:, 0], ev3[:, 1], nn_perfs, 'ko')
    ax.set_title('Eigenvalue 3')
    plt.xlabel('Real')
    plt.ylabel('Imag')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(ev1_mod, ev2_mod, nn_perfs, 'go')    
    plt.xlabel('|e1|')
    plt.ylabel('|e2|')
        
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(ev1[:, 0], ev2[:, 0], nn_perfs, 'ro')    
    plt.xlabel('real(e1)')
    plt.ylabel('real(e2)')

    plt.show()

def plot_stim_pca(stim_file):
    
    (stims, stim_proj, class_indx) = stim_pca(stim_file)
    
    clrs = ['r', 'g', 'b', 'k', 'c']
    stim_classes = np.unique(class_indx)
    
    plt.clf()
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for k,sc in enumerate(stim_classes):
        sc_indx = class_indx == sc
        ax.scatter(stim_proj[sc_indx, 0], stim_proj[sc_indx, 1], stim_proj[sc_indx, 2], c=clrs[sc])       
    ax.set_title('3D PCA Projected Stimuli')
        
    plt.show()

def plot_pseudospectra_by_perf(pdata, rootdir='/home/cheese63/git/prorn/data'):
    
    pdata = filter_perfs(pdata)
    
    num_plots = 25
    
    indx_off = [0, len(pdata)-num_plots]
    pdata.sort(key=operator.attrgetter('nn_perf'))
    weights = [[], []]
    for k,offset in enumerate(indx_off):
        pend = offset + num_plots
        for m,p in enumerate(pdata[offset:pend]):
            fname = os.path.join(rootdir, p.file_name)
            net_key = p.net_key
            print 'k=%d, offset=%d, pdata[%d] (file_name=%s, net_key=%s)' % (k, offset, m,  p.file_name, net_key)
            f = h5py.File(fname, 'r')
            W = np.array(f[net_key]['W'])
            weights[k].append(W)        
            f.close()
    
    perrow = int(np.sqrt(num_plots))
    percol = perrow
    
    for j,offset in enumerate(indx_off):
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)    
        for k in range(num_plots):
            W = weights[j][k]
            ax = fig.add_subplot(perrow, percol, k)
            plot_pseudospectra(W, bounds=[-3, 3, -3, 3], npts=50, ax=ax, colorbar=False, log=False)
            plt.axhline(0.0, color='k', axes=ax)
            plt.axvline(0.0, color='k', axes=ax)
            plt.xticks([], [])
            plt.yticks([], [])
            
            p = pdata[offset + k]
            for m in range(3):
                ev = p.eigen_values[m]
                ax.plot(ev.real, ev.imag, 'ko', markerfacecolor='w')
        if offset == 0:
            plt.suptitle('Pseudospectra of Top %d Networks' % num_plots)
        else:
            plt.suptitle('Pseudospectra of Bottom %d Networks' % num_plots)
            
    plt.show()
    
    

def save_to_png(fig, output_file):
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(output_file, dpi=72)
    