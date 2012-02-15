import os
import time
import operator

import pylab
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg
from prorn.network import EchoStateNetwork
from prorn.sim import run_sim

try:
    import mayavi.mlab as mlab
except:
    import enthought.mayavi.mlab as mlab

import h5py

from prorn.config import *
from prorn.info import fisher_memory_matrix
from prorn.stim import StimPrototype, stim_pca
from prorn.readout import get_samples
from prorn.spectra import plot_pseudospectra, plot_pseudospectra_contour,\
    compute_pseudospectra
from prorn.analysis import top100_weight_pca, get_top_perfs

def plot_trajectory(perf_data, stimset, stim_class, symbols):
        
    net = perf_data.net
    net.noise_std = 0.0
        
        
    mlab.figure(bgcolor=(0.5, 0.5, 0.5), fgcolor=(0.0, 0.0, 0.0))
    for symbol in symbols:
        stim_md5 = stimset.symbol_to_md5[symbol][0]
        stim = stimset.all_stims[stim_md5]
        net_sims = run_sim(net, {stim_md5:stim},
                           burn_time=100,
                           pre_stim_time = 1,
                           post_stim_time=25,
                           num_trials=1)
        
        sim = net_sims[stim_md5]
        avg_resp = sim.responses[0, :, :].squeeze()
        stim_start = 0
        stim_end = len(stim) + 1
        
        stimsym = stimset.md5_to_symbol[stim_md5]
        stim_str = '%s:%s (%0.2f)' % (stimsym, ''.join(['%d' % s for s in stim]), perf_data.logit_perf)
    
        t = np.arange(0, stim_end)
        traj = mlab.plot3d(avg_resp[0:stim_end, 0], avg_resp[0:stim_end, 1], avg_resp[0:stim_end, 2], t,
                           colormap='hot', tube_radius=None)
        #mlab.points3d(avg_resp[stim_start, 0], avg_resp[stim_start, 1], avg_resp[stim_start, 2], scale_factor=0.300)
        mlab.points3d(avg_resp[stim_end-1, 0], avg_resp[stim_end-1, 1], avg_resp[stim_end-1, 2], scale_factor=0.900)
    
    #mlab.colorbar()
    #mlab.title(stim_str)
    mlab.title('%0.2f' % perf_data.logit_perf)
    

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
  
      
def plot_single_net_and_stim_movie(stim_file, net_file, net_key, stim_key, trial, output_file):
    
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
    fig_path = os.path.join(TEMP_DIR, '%s.png' % fig_prefix)
    
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
    
    rm_cmd = 'rm %s' % os.path.join(TEMP_DIR, fig_prefix_rm)
    print 'Removing temp files...'
    os.system(rm_cmd)
    

def create_net_and_stim_movies(stim_file, net_file, output_dir='/home/cheese63/net_movies', limit=None):
    
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
                plot_single_net_and_stim_movie(stim_file, net_file, net_key, stim_key, trial, TEMP_DIR, output_file)
                
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
    

def plot_perf_histograms(pdata):
    
    logit_perfs = np.array([p.logit_perf for p in pdata])
    
    plt.clf()
    
    fig = plt.gcf()    
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(logit_perfs, bins=25)
    ax.set_title('% Correct (MNLogit Readout)')
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
     

def plot_top_100_weights(pdata):
        
    weights = []
    for p in pdata[0:100]:
        fname = os.path.join(DATA_DIR, p.file_name)
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

def plot_pseudospectra_by_perf(pdata, perf_attr='logit_perf', contour=False, levels=None):
    
    num_plots = 25
    
    indx_off = [0, len(pdata)-num_plots]
    weights = [[], []]
    for k,offset in enumerate(indx_off):
        pend = offset + num_plots
        for m,p in enumerate(pdata[offset:pend]):
            weights[k].append(p.W)        
            
    perrow = int(np.sqrt(num_plots))
    percol = perrow
    
    for j,offset in enumerate(indx_off):
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)    
        for k in range(num_plots):
            W = weights[j][k]
            N = W.shape[0]
            ax = fig.add_subplot(perrow, percol, k)
            if contour:
                plot_pseudospectra_contour(W, bounds=[-3, 3, -3, 3], npts=50, ax=ax, colorbar=False, invert=False, log=False, levels=levels)
            else:
                plot_pseudospectra(W, bounds=[-3, 3, -3, 3], npts=50, ax=ax, colorbar=True, log=True)
            plt.axhline(0.0, color='k', axes=ax)
            plt.axvline(0.0, color='k', axes=ax)
            
            cir = pylab.Circle((0.0, 0.0), radius=1.00,  fc='gray', fill=False)
            pylab.gca().add_patch(cir)
            
            plt.xticks([], [])
            plt.yticks([], [])
            
            if not contour:
                p = pdata[offset + k]
                for m in range(N):
                    ev = p.eigen_values[m]
                    ax.plot(ev.real, ev.imag, 'ko', markerfacecolor='w')
        if offset == 0:
            plt.suptitle('Pseudospectra of Top %d Networks' % num_plots)
        else:
            plt.suptitle('Pseudospectra of Bottom %d Networks' % num_plots)
            
    plt.show()

def plot_smin_hist_by_perf(pdata, tau=1):
    
    num_plots = 9
    
    indx_off = [0, len(pdata)-num_plots]
    weights = [[], []]
    for k,offset in enumerate(indx_off):
        pend = offset + num_plots
        for m,p in enumerate(pdata[offset:pend]):
            weights[k].append(p.W)        
            
    perrow = int(np.sqrt(num_plots))
    percol = perrow
    
    for j,offset in enumerate(indx_off):
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)    
        for k in range(num_plots):
            W = weights[j][k]
            N = W.shape[0]
            ax = fig.add_subplot(perrow, percol, k)
            (X, Y, Z, smin) = compute_pseudospectra(W, bounds=[-3, 3, -3, 3], npts=50, invert=False)
            
            a = np.real(Z)
            K = smin / a
            eatau = np.exp(a*tau) 
            maxg = eatau / (1.0 + ((eatau - 1.0) / K))

            ax.hist(maxg[(a > 0) * (K > 1)].ravel(), bins=20)
            #ax.hist(smin.ravel(), bins=20)
            
    plt.show()
    

def plot_lower_bound_perf(pdata, tau=1, indicies=None):
    
    
    if indicies is None:
        indicies = range(len(pdata))
    
    lb = []
    perfs = []
    
    for k in indicies:
        p = pdata[k]    
        W = p.W
        (X, Y, Z, smin) = compute_pseudospectra(W, bounds=[-3, 3, -3, 3], npts=50, invert=True)
        
        a = np.real(Z)
        K = smin / a
        eatau = np.exp(a*tau) 
        maxg = eatau / (1.0 + ((eatau - 1.0) / K))
        
        lb.append(maxg[(a > 0) * (K > 1)].ravel().mean())
        perfs.append(p.logit_perf)
            
    plt.plot(lb, perfs, 'go')    
    plt.show()    

    
def plot_abscissa_eps_by_perf(pdata, perf_attr='logit_perf', eps=0.5, indicies=None, mean=False, abs=False, right_half_only=False):
        
    if indicies is None:
        indicies = range(len(pdata))
    perfs = []
    abscissas = []
    for k in indicies:
        p = pdata[k]
        (X, Y, Z, smin) = compute_pseudospectra(p.W, bounds=[-3, 3, -3, 3], npts=50, invert=False)
        zvals = Z[smin < eps]
        if zvals.shape[0] > 0:
            zreal = np.array([a.real for a in zvals])
            if right_half_only:
                zreal = zreal[zreal > 0]
            if abs:
                zreal = np.abs(zreal)
            if len(zreal) > 0:
                if mean:
                    abscissas.append(zreal.mean())
                else:
                    abscissas.append(zreal.max())
                perfs.append(p.logit_perf)    
    
    plt.plot(abscissas, perfs, 'bo')
    plt.show()

    
def plot_perf_by_schur_offdiag(pdata, perf_attr='logit_perf'):
    
    perfs = []
    schur_offdiag_sum = []
    for m,p in enumerate(pdata):
        fname = os.path.join(DATA_DIR, p.file_name)
        net_key = p.net_key
        f = h5py.File(fname, 'r')
        W = np.array(f[net_key]['W'])
        f.close()
        (T, Z) = scipy.linalg.schur(W, 'complex')
        M = np.abs(T)
        M -= np.diag(np.diag(M))
    
        perfs.append(getattr(p, perf_attr))
        schur_offdiag_sum.append(M.sum())
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(schur_offdiag_sum, perfs, 'ko')
    ax.set_title('Sum of off-diagnonal elements of Schur decomposition')
    plt.ylabel('Performance')
    plt.xlabel('Sum')
    plt.show()

def plot_commutivity(pdata, perf_attr='logit_perf'):
    
    num_plots = 25
    
    indx_off = [0, len(pdata)-num_plots]
    weights = [[], []]
    comm = [[], []]
    for k,offset in enumerate(indx_off):
        pend = offset + num_plots
        for m,p in enumerate(pdata[offset:pend]):
            weights[k].append(p.W)
            W = np.matrix(p.W)
            Wt = np.matrix(np.transpose(p.W))
            c1 = W*Wt
            c2 = Wt*W
            cdiff = (c1 - c2).sum()
            comm[k].append(cdiff)
    
    top_cdiff = np.array(comm[0])
    bottom_cdiff = np.array(comm[1])
    
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.hist(top_cdiff, bins=20)
    ax.set_title('Commutivity Differences: Top 25')
    ax = fig.add_subplot(2, 1, 2)    
    ax.hist(bottom_cdiff, bins=20)
    ax.set_title('Commutivity Differences: Bottom 25')
        
    

def plot_schur_by_perf(pdata, perf_attr='logit_perf', show_unitary=False):
    
    num_plots = 25
    
    indx_off = [0, len(pdata)-num_plots]
    weights = [[], []]
    comm = [[], []]
    for k,offset in enumerate(indx_off):
        pend = offset + num_plots
        for m,p in enumerate(pdata[offset:pend]):
            weights[k].append(p.W)
            W = np.matrix(p.W)
            Wt = np.matrix(np.transpose(p.W))
            c1 = W*Wt
            c2 = Wt*W
            cdiff = (c1 - c2).sum()
            comm[k].append(cdiff)
    
    perrow = int(np.sqrt(num_plots))
    percol = perrow
    
    for j,offset in enumerate(indx_off):
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)    
        for k in range(num_plots):
            W = weights[j][k]
            (T, Z) = scipy.linalg.schur(W, 'complex')
            if show_unitary:
                M = np.abs(np.transpose(Z))
            else:
                M = np.abs(T)
            
            ax = fig.add_subplot(perrow, percol, k)
            ax.imshow(M, interpolation='nearest', vmin=-1.0, vmax=1.0, cmap=cm.jet)
            ax.set_title('%0.2f' % comm[j][k])            
            plt.xticks([], [])
            plt.yticks([], [])
            
            p = pdata[offset + k]
            
        if offset == 0:
            plt.suptitle('Schur Decomposition of Top %d Networks' % num_plots)
        else:
            plt.suptitle('Schur Decomposition of Bottom %d Networks' % num_plots)
            
    plt.show()
    
    
def plot_input_by_schur(pdata, perf_attr='logit_perf', weight_transform=False):
    
    num_plots = 25
    
    indx_off = [0, len(pdata)-num_plots]
    weights = [[], []]
    win = [[], []]
    for k,offset in enumerate(indx_off):
        pend = offset + num_plots
        for m,p in enumerate(pdata[offset:pend]):
            weights[k].append(p.W)
            win[k].append(p.Win)
            
    perrow = int(np.sqrt(num_plots))
    percol = perrow
    
    for j,offset in enumerate(indx_off):
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)    
        for k in range(num_plots):
            W = weights[j][k]
            Win = win[j][k].reshape(W.shape[0], 1)
            (T, Z) = scipy.linalg.schur(W, 'complex')
                        
            c = np.matrix(Z.conjugate()) * Win
            if weight_transform:
                c = np.matrix(T) * c

            xy = np.array([(z.real, z.imag) for z in np.array(c).squeeze()])
            
            num_real = len( (xy[:, 1] == 0).nonzero()[0] )
            
            maxval = (np.abs(xy).max())*1.1
            
            ax = fig.add_subplot(perrow, percol, k)
            ax.plot(xy[:, 0], xy[:, 1], 'ro')            
            ax.set_title('%d' % num_real)
            plt.axhline(0.0, color='k', axes=ax)
            plt.axvline(0.0, color='k', axes=ax)
            
            cir = pylab.Circle((0.0, 0.0), radius=1.00,  fc='gray', fill=False)
            pylab.gca().add_patch(cir)            
                        
            plt.xticks([], [])
            plt.yticks([], [])
            plt.axis([-maxval, maxval, -maxval, maxval])
            
            p = pdata[offset + k]
            
        if offset == 0:
            plt.suptitle('Schur-projected Input of Top %d Networks' % num_plots)
        else:
            plt.suptitle('Schur-projected Input of Bottom %d Networks' % num_plots)
            
    plt.show()

def plot_input_by_schur_nreal(pdata):
    
    nreal = []
    perf = []
    for m,p in enumerate(pdata):        
        W = p.W
        Win = p.Win.reshape([W.shape[0], 1])
        (T, Z) = scipy.linalg.schur(W, 'complex')
                        
        c = np.matrix(Z.conjugate()) * Win                         
        xy = np.array([(z.real, z.imag) for z in np.array(c).squeeze()])
        
        num_real = len( (xy[:, 1] == 0).nonzero()[0] )
        nreal.append(num_real)
        perf.append(p.logit_perf)
    
    plt.plot(nreal, perf, 'go')
    plt.title('# of real valued projects vs performance')            
    plt.show()
    
def plot_fmm_by_perf(pdata, perf_attr='logit_perf'):
    
    num_plots = 25
    
    indx_off = [0, len(pdata)-num_plots]
    weights = [[], []]
    inputs = [[], []]
    perfs = [[], []] 
    for k,offset in enumerate(indx_off):
        pend = offset + num_plots
        for m,p in enumerate(pdata[offset:pend]):
            weights[k].append(p.W)
            inputs[k].append(p.Win)
            perfs[k].append(p.logit_perf)
    
    perrow = int(np.sqrt(num_plots))
    percol = perrow
    
    for j,offset in enumerate(indx_off):
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)    
        for k in range(num_plots):
            W = weights[j][k]
            v = inputs[j][k]
            J = fisher_memory_matrix(W, v, npts=15)
            Jlog = np.log(J+1.0)
            
            ax = fig.add_subplot(perrow, percol, k)
            ax.imshow(J, interpolation='nearest', cmap=cm.jet)
            #ax.set_title('%0.2f' % perfs[j][k])            
            plt.xticks([], [])
            plt.yticks([], [])
            
            p = pdata[offset + k]
            
        if offset == 0:
            plt.suptitle('Fisher Memory Matricies of Top %d Networks' % num_plots)
        else:
            plt.suptitle('Fisher Memory Matricies of Bottom %d Networks' % num_plots)
            
    plt.show()    

def plot_fmc_by_perf(pdata, perf_attr='logit_perf'):
    
    num_plots = 25
    
    indx_off = [0, len(pdata)-num_plots]
    weights = [[], []]
    inputs = [[], []]
    for k,offset in enumerate(indx_off):
        pend = offset + num_plots
        for m,p in enumerate(pdata[offset:pend]):
            weights[k].append(p.W)
            inputs[k].append(p.Win)            
    
    perrow = int(np.sqrt(num_plots))
    percol = perrow
    
    for j,offset in enumerate(indx_off):
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)    
        for k in range(num_plots):
            W = weights[j][k]
            v = inputs[j][k]
            J = fisher_memory_matrix(W, v)
            fmc = np.diag(J)
            fmc /= fmc.max()
            
            ax = fig.add_subplot(perrow, percol, k)
            ax.plot(fmc, 'k-')
            ax.set_title('%0.3f' % fmc.sum())            
            plt.xticks([], [])
            plt.yticks([], [])
            
            p = pdata[offset + k]
            
        if offset == 0:
            plt.suptitle('Fisher Memory Curves of Top %d Networks' % num_plots)
        else:
            plt.suptitle('Fisher Memory Curves of Bottom %d Networks' % num_plots)
            
    plt.show()    
    

def plot_perf_by_jtot(pdata, perf_attr='logit_perf'):
    
    perfs = []
    jtots = []
    for m,p in enumerate(pdata):
        J = fisher_memory_matrix(p.W, p.Win)
        fmc = np.diag(J)
        #fmc /= fmc.max()
        perfs.append(getattr(p, perf_attr))
        jtots.append(fmc.sum())
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(jtots, perfs, 'ko')
    ax.set_title('Jtot vs Performance')
    plt.xlabel('Jtot')
    plt.ylabel('Performance')
    plt.show()
    
def plot_perf_by_eigsum(pdata, perf_attr='logit_perf'):
    
    perfs = []
    eigsum = []
    for m,p in enumerate(pdata):
        eigsum.append(np.abs(p.eigen_values).sum())
        perfs.append(p.logit_perf)
            
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(eigsum, perfs, 'ko')
    ax.set_title('Eigsum vs Performance')
    plt.xlabel('Eigsum')
    plt.ylabel('Performance')
    plt.show()
    
def plot_perf_by_abscissa(pdata, perf_attr='logit_perf', mean=False):
    
    perfs = []
    aval = []
    for m,p in enumerate(pdata):
        rv = np.array([l.real for l in p.eigen_values])
        if not mean:
            aval.append(rv.max())
        else:
            aval.append(rv.mean())            
        perfs.append(p.logit_perf)
            
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(aval, perfs, 'ko')
    ax.set_title('Abscissa vs Performance')
    plt.xlabel('Abscissa')
    plt.ylabel('Performance')
    plt.show()
    

def plot_perf_by_jsum(pdata, perf_attr='logit_perf'):
    
    perfs = []
    jtots = []
    for m,p in enumerate(pdata):
        fname = os.path.join(DATA_DIR, p.file_name)
        net_key = p.net_key
        f = h5py.File(fname, 'r')
        W = np.array(f[net_key]['W'])
        Win = np.array(f[net_key]['Win']).squeeze()
        v = Win.squeeze()
        f.close()
        J = fisher_memory_matrix(W, v)
        jusum = np.abs(J[np.triu_indices(len(J))]).sum()
        perfs.append(getattr(p, perf_attr))
        jtots.append(jusum)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(jtots, perfs, 'ko')
    ax.set_title('Upper Triangular Sum of J vs Performance')
    plt.xlabel('UT Sum')
    plt.ylabel('Performance')
    plt.show()

def plot_perf_hists(top_net_files):
    
    for k,tfile in enumerate(top_net_files):
        (rdir, fname) = os.path.split(tfile)
        pdata = get_top_perfs(tfile)
        perfs = np.array([p.logit_perf for p in pdata])
        
        plt.subplot(len(top_net_files), 1, k+1)
        plt.hist(perfs, bins=20)
        plt.title('%s: mean=%0.2f +/- %0.2f' % (fname, perfs.mean(), perfs.std()))
        plt.xlim((0, 1.00))
    
    plt.show()

def save_to_png(fig, output_file):
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(output_file, dpi=72)
