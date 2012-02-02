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

from mvpa2.datasets import dataset_wizard
from mvpa2.clfs.smlr import SMLR

from prorn.stim import stim_pca

def get_samples(net_file, net_key, exp_name, stim_keys, readout_time):
    """ Sample from the state of a recurrent net in an hdf5 flie. The
        readout time is given relative to the end of the stimulus, so
        that readout_time=0 corresponds to the last time point in the
        stimulus.
    """
    
    samps = {}
    
    f = h5py.File(net_file, 'r')
    resp_grp = f[net_key][exp_name]['responses']
    
    for stim_key in stim_keys:
        
        samps[stim_key] = []
    
        stim_grp = resp_grp[stim_key]
        stim_start = int(stim_grp.attrs['stim_start'])
        stim_end = int(stim_grp.attrs['stim_end'])
        num_trials = int(stim_grp.attrs['num_trials'])
        resps = np.array(stim_grp['responses'])
        
        t_readout = stim_end + readout_time
        
        for n in range(num_trials):
            net_state = resps[n, t_readout, :].squeeze()
            samps[stim_key].append(net_state)
    
    f.close()
    return samps

def to_mvpa_dataset(stimset, samples):
    ds_data = []
    targets = []
    for stim_key,samps in samples.iteritems():
        for samp in samps:
            sym = stimset.md5_to_symbol[stim_key]
            targets.append(sym)
            ds_data.append(samp)
    ds_data = np.array(ds_data)
    
    train_len = int(0.75*ds_data.shape[0])
    ds_indx = range(ds_data.shape[0])
    np.random.shuffle(ds_indx)
    train_index = ds_indx[:train_len]
    valid_index = ds_indx[train_len:]
    
    ds_train = dataset_wizard(ds_data[train_index, :], targets=np.array(targets)[train_index])
    ds_valid = dataset_wizard(ds_data[valid_index, :], targets=np.array(targets)[valid_index])
    
    return (ds_train, ds_valid)

def train_readout_mnlogit(stimset, samples):
    (ds_train, ds_valid) = to_mvpa_dataset(stimset, samples)
    clf = SMLR()
    
    clf.train(ds_train)
    
    preds = clf.predict(ds_valid)
    actual = ds_valid.sa['targets']
    zeq = np.array([a == p for (a,p) in zip(actual, preds)])
    nc = float(len((zeq == True).nonzero()[0])) 
    print '%d correct out of %d' % (nc, len(preds))
    percent_correct = nc / float(len(preds))
    print 'Percent Correct: %0.3f' % percent_correct
    
    return (ds_train, ds_valid, clf, actual, preds)


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
