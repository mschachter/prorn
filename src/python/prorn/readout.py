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

def get_net_performance(net_file, exp_name, stim_class, perf_type='mnlogit'):
    
    perfs = {}
    f = h5py.File(net_file, 'r')    
    for net_key in f.keys():
        perf = float(f[net_key][exp_name]['performance'][stim_class][perf_type][()])
        perfs[net_key] = perf
    return perfs
    

def compute_performance_per_net(net_file, exp_name, stimset, stim_class, readout_time):
    
    stim_keys = stimset.class_to_md5[stim_class]
    f = h5py.File(net_file, 'a')
    
    for net_key in f.keys():        
        samps = get_samples(net_file, net_key, exp_name, stim_keys, readout_time)
        mn_perf = train_readout_mnlogit(stimset, samps)
        l_perf = train_readout_logit(stimset, samps)
        
        if 'performance' not in f[net_key][exp_name]:
            f[net_key][exp_name].create_group('performance')
        perf_grp = f[net_key][exp_name]['performance']
        if stim_class not in perf_grp:
            perf_grp.create_group(stim_class)
        stim_grp = perf_grp[stim_class]
        stim_grp['mnlogit'] = mn_perf
        stim_grp['logit'] = l_perf
        
    f.close()
    

def combine_samples(net_file, net_keys, exp_name, stim_keys, readout_time):
    
    samps = {}
    
    #get individual samples from each net
    individual_samps = []
    for net_key in net_keys:
        isamps = get_samples(net_file, net_key, exp_name, stim_keys, readout_time)
        individual_samps.append(isamps)
    
    #assuming each experiment was run with an identical random seed
    #for each network, and the number of trials is the same for each
    #stimulus, concatenate all the samples
    
    concat_samps = {}
    for stim_key in individual_samps[0].keys():
        all_samps = []
        
        num_trials = -1
        for k,isamps in enumerate(individual_samps):
            net_samps = isamps[stim_key]
            num_trials = len(net_samps)
            all_samps.append(net_samps)
            
        concat_samps[stim_key] = []
        for n in range(num_trials):
            samp = []
            for k,net_samps in enumerate(all_samps):
                samp.extend(net_samps[n])
            concat_samps[stim_key].append(samp)
        
    return concat_samps
    

def get_samples(net_file, net_key, exp_name, stim_keys, readout_time):
    """ Sample from the state of a recurrent net in an hdf5 file. The
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
        sym = stimset.md5_to_symbol[stim_key]
        for samp in samps:
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
    #print '%d correct out of %d' % (nc, len(preds))
    percent_correct = nc / float(len(preds))
    #print 'SMLogit Percent Correct: %0.3f' % percent_correct
    
    return percent_correct

def get_np_dataset(stimset, samples):
    
    sym_to_index = {}
    index_to_sym = {}
    for k,sym in enumerate(stimset.symbol_to_md5.keys()):
        sym_to_index[sym] = k
        index_to_sym[k] = sym
    
    data = []
    for stim_key,samps in samples.iteritems():
        for samp in samps:
            row = []
            sym = stimset.md5_to_symbol[stim_key]
            sym_index = sym_to_index[sym]
            row.extend(samp)
            row.append(sym_index)
            data.append(row)
    data = np.array(data)
    np.random.shuffle(data)
    
    ntrain = int(0.75*data.shape[0])
    train_data = data[:ntrain, :]
    test_data = data[ntrain:, :]
    
    return (train_data, test_data, index_to_sym)

def train_readout_logit(stimset, samples):
    
    (train_data, test_data, index_to_sym) = get_np_dataset(stimset, samples)
        
    N = train_data.shape[1]-1
    logr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1)
    logr.fit(train_data[:, :N], train_data[:, -1])
        
    test_pred = logr.predict(test_data[:, 0:N])
    pred_diff = test_pred - test_data[:, -1]
    percent_correct = len((pred_diff == 0).nonzero()[0]) / float(len(test_pred))
    print 'Logit Percent correct: %0.3f' % percent_correct
    return percent_correct
        

def train_readout_logit_onevsall(stimset, samples):
    
    (train_data, test_data, index_to_sym) = get_np_dataset(stimset, samples)    
    models = {}
    
    for sym_index,sym in index_to_sym.iteritems():
        
        #train a logistic regression model on just this stim class vs. the rest
        mdata = copy.deepcopy(data)
        data[data[:, -1] != sym_index, -1] = 0
        data[data[:, -1] == sym_index, -1] = 1
        print ''
        print list(data)
        print ''
        
        N = data.shape[1]-1
        
        logr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1)
        logr.fit(train_data[:, :N], train_data[:, -1])
        
        test_pred = logr.predict(test_data[:, 0:N])
        pred_diff = np.abs(test_pred - test_data[:, -1])
        zero_one_loss = pred_diff.sum() / test_data.shape[0]
        print 'Stim class %s loss: %0.3f' % (sym, zero_one_loss)
        
        models[sym] = logr
        
    
    
    
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
