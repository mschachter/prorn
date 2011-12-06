import numpy as np

import h5py

from scikits.learn.linear_model import LogisticRegression
from scikits.learn.svm import SVC
from scikits.statsmodels.discrete.discrete_model import MNLogit

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, TanhLayer

from prorn.stim import stim_pca

def get_samples(net_file, net_key, readout_window):
    
    Ninternal = -1
    
    samps = []
    all_stim_classes = []
    
    f = h5py.File(net_file, 'r')
    stim_keys = [key for key in f[net_key].keys() if len(key) > 4 and key[0:4] == 'stim']
    for stim_key in stim_keys:
        net_stims = np.array(f[net_key][stim_key])
        stim_start = f[net_key][stim_key].attrs['stim_start']
        stim_end = f[net_key][stim_key].attrs['stim_end']
        
        stim_class = int(stim_key[5:])
        if stim_class not in all_stim_classes:
            all_stim_classes.append(stim_class)
        
        if Ninternal == -1:
            Ninternal = net_stims.shape[2]
        if net_stims.shape[2] != Ninternal:
            print 'Wonky data size, # neurons for net_key=%s, stim_key=%s doesn\'t match others: %d != %d' % \
                  (net_key, stim_key, net_stims.shape[2], Ninternal)
        
        ntrials = net_stims.shape[0]
        for k in range(ntrials):
            net_state = net_stims[k, :, :].squeeze()
            #extract time points for readout
            readout_times = stim_end + readout_window
            for rt in readout_times:
                readout_state = net_state[rt, :].squeeze()
                samps.append((readout_state, stim_class))
    f.close()        
    np.random.shuffle(samps)
    return (samps, all_stim_classes, Ninternal)


def train_readout_nn(net_file, net_key, readout_window=np.arange(0, 1), num_hidden=2):
        
    (samps, all_stim_classes, Ninternal) = get_samples(net_file, net_key, readout_window)
    
    alldata = ClassificationDataSet(Ninternal, 1, nb_classes=len(all_stim_classes))
    for readout_state,stim_class in samps:
        alldata.addSample(readout_state, [stim_class])
            
    tstdata, trndata = alldata.splitWithProportion( 0.25 )
    
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()
    
    fnn = buildNetwork( trndata.indim, num_hidden, trndata.outdim, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=False, weightdecay=0.01)
    
    test_errors = []
    num_slope_samps = 10
    slope = 0.0
    
    while True:
        
        if len(test_errors) >= num_slope_samps:
            coef = np.polyfit(np.arange(num_slope_samps), test_errors[-num_slope_samps:], 1)
            slope = coef[0]
            if slope > 0.0:
                print 'Test error slope > 0.0, stopping'
                break
        
        trainer.train()        
        train_err = percentError( trainer.testOnClassData(), trndata['class'])
        test_err = percentError( trainer.testOnClassData(dataset=tstdata), tstdata['class'])
        print "Iteration: %d, train_err=%0.4f, test_err=%0.4f, slope=%0.4f" % (trainer.totalepochs, train_err, test_err, slope)
        test_errors.append(test_err)
        
    return (train_err, test_err, fnn, trainer)



def train_readout_logit(net_file, net_key, readout_window=np.arange(0, 1)):
    
    (samps, all_stim_classes, Ninternal) = get_samples(net_file, net_key, readout_window)
    
    losses = []
    
    for stim_class in all_stim_classes:
        
        data = np.zeros([len(samps), Ninternal+1])
        for k,(state,samp_sc) in enumerate(samps):
            data[k, 0:Ninternal] = state
            data[k, -1] = int(samp_sc == stim_class)
        
        ntrain = int(0.75*len(samps))
        ntest =  len(samps) - ntrain
        train_data = data[0:ntrain, :]
        test_data = data[ntrain:, :]
        
        logr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1)
        logr.fit(train_data[:, 0:Ninternal], train_data[:, -1])
        
        test_pred = logr.predict(test_data[:, 0:Ninternal])
        pred_diff = np.abs(test_pred - test_data[:, -1])
        zero_one_loss = pred_diff.sum() / test_data.shape[0]
        losses.append(zero_one_loss) 
        
        #print 'Stim class %d loss: %0.3f' % (stim_class, zero_one_loss)
        
    losses = np.array(losses)
    return (losses.mean(), losses.std())
        
        
def train_readout_svm(net_file, net_key, readout_window=np.arange(3, 10)):
    
    (samps, all_stim_classes, Ninternal) = get_samples(net_file, net_key, readout_window)
    
    losses = []
    
    for stim_class in all_stim_classes:
        
        data = np.zeros([len(samps), Ninternal+1])
        for k,(state,samp_sc) in enumerate(samps):
            data[k, 0:Ninternal] = state
            data[k, -1] = int(samp_sc == stim_class)
        
        ntrain = int(0.75*len(samps))
        ntest =  len(samps) - ntrain
        train_data = data[0:ntrain, :]
        test_data = data[ntrain:, :]
        
        svc = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001) 
        svc.fit(train_data[:, 0:Ninternal], train_data[:, -1])
        
        test_pred = svc.predict(test_data[:, 0:Ninternal])
        pred_diff = np.abs(test_pred - test_data[:, -1])
        zero_one_loss = pred_diff.sum() / test_data.shape[0]
        losses.append(zero_one_loss)
        #print 'Stim class %d loss: %0.3f' % (stim_class, zero_one_loss)
                
    losses = np.array(losses)
    return (losses.mean(), losses.std())


def train_pca_stim_nn(stim_file):
    
    (stims, stim_proj, class_index) = stim_pca(stim_file)
    stim_classes = np.unique(class_index)
    
    alldata = ClassificationDataSet(3, 1, nb_classes=len(stim_classes))
    nsamps = stim_proj.shape[0]
    for k in range(nsamps):
        alldata.addSample(stim_proj[k, :].squeeze(), class_index[k])
            
    tstdata, trndata = alldata.splitWithProportion( 0.25 )
    
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()
    
    fnn = buildNetwork( trndata.indim, 2, trndata.outdim, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=False, weightdecay=0.01)
    
    test_errors = []
    num_slope_samps = 10
    slope = 0.0
    
    while True:
        
        if len(test_errors) >= num_slope_samps:
            coef = np.polyfit(np.arange(num_slope_samps), test_errors[-num_slope_samps:], 1)
            slope = coef[0]
            if slope > 0.0:
                print 'Test error slope > 0.0, stopping'
                break
        
        trainer.train()        
        train_err = percentError( trainer.testOnClassData(), trndata['class'])
        test_err = percentError( trainer.testOnClassData(dataset=tstdata), tstdata['class'])
        print "Iteration: %d, train_err=%0.4f, test_err=%0.4f, slope=%0.4f" % (trainer.totalepochs, train_err, test_err, slope)
        test_errors.append(test_err)
        
    return (train_err, test_err, fnn, trainer)
    
        
    
        