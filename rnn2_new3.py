#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:24:52 2017

@author: shaks
"""

import os, sys
import argparse
import time
import itertools
import cPickle
import logging
import random
import string
import preprocess

from collections import defaultdict
from random import shuffle


import numpy as np
import tensorflow as tf    
import matplotlib.pyplot as plt

from model import Model, NottinghamModel

def get_config_name(config):
    def replace_dot(s): return s.replace(".", "p")
    return "numLayers_" + str(config.num_layers) + \
           "_hidSize_" + str(config.hidden_size) + \
            replace_dot("_melCoef_{}".format(config.melody_coeff)) + \
            replace_dot("_dropOut_{}".format(config.dropout_prob)) + \
            replace_dot("_inpDropOut_{}".format(config.input_dropout_prob)) + \
            replace_dot("_timeBatchLen_{}".format(config.time_batch_len))+ \
             replace_dot("_cellType_{}".format(config.cell_type))

class DefaultConfig(object):
    # model parameters
    num_layers = 2
    hidden_size = 200
    melody_coeff = 0.5
    dropout_prob = 0.5
    input_dropout_prob = 0.8
    cell_type = 'LSTM'

    # learning parameters
    max_time_batches = 10
    time_batch_len = 128
    learning_rate = 1e-3
    learning_rate_decay = 0.9
    num_epochs = 250

    # metadata
    dataset = 'softmax'
    model_file = ''

    def __repr__(self):
        return """Num Layers: {}, Hidden Size: {}, Melody Coeff: {}, Dropout Prob: {}, Input Dropout Prob: {}, Cell Type: {}, Time Batch Len: {}, Learning Rate: {}, Decay: {}""".format(self.num_layers, self.hidden_size, self.melody_coeff, self.dropout_prob, self.input_dropout_prob, self.cell_type, self.time_batch_len, self.learning_rate, self.learning_rate_decay)
data = []
targets = []    

def run_epoch(session, model, batches, training=False, testing=False):

    # shuffle batches
    shuffle(batches)

    target_tensors = [model.loss, model.final_state]
    if testing:
        target_tensors.append(model.probs)
        batch_probs = defaultdict(list)
    if training:
        target_tensors.append(model.train_step)

    losses = []
    for data, targets in batches:
        # save state over unrolling time steps
        batch_size = data[0].shape[1]
        num_time_steps = len(data)
        state = model.get_cell_zero_state(session, batch_size) 
        probs = list()

        for tb_data, tb_targets in zip(data, targets):
            if testing:
                tbd = tb_data
                tbt = tb_targets
            else:
                # shuffle all the batches of input, state, and target
                batches = tb_data.shape[1]
                permutations = np.random.permutation(batches)
                tbd = np.zeros_like(tb_data)
                tbd[:, np.arange(batches), :] = tb_data[:, permutations, :]
                tbt = np.zeros_like(tb_targets)
                tbt[:, np.arange(batches), :] = tb_targets[:, permutations, :]
                state[np.arange(batches)] = state[permutations]

            feed_dict = {
                model.initial_state: state,
                model.seq_input: tbd,
                model.seq_targets: tbt,
            }
            results = session.run(target_tensors, feed_dict=feed_dict)

            losses.append(results[0])
            state = results[1]
            if testing:
                batch_probs[num_time_steps].append(results[2])

    loss = sum(losses) / len(losses)

    if testing:
        return [loss, batch_probs]
    else:
        return loss


def batch_data(seqIn, num_time_steps):

    seq = [s[:(num_time_steps*time_batch_len)+1, :] for s in seqIn]

    # stack sequences depth wise (along third axis).
    stacked = np.dstack(seq)
    # swap axes so that shape is (SEQ_LENGTH X BATCH_SIZE X INPUT_DIM)
    data = np.swapaxes(stacked, 1, 2)
    # roll data -1 along lenth of sequence for next sequence prediction
    targets = np.roll(data, -1, axis=0)
    # cutoff final time step, cut of count from targets
    data = data[:-1, :, :]
    targets = targets[:-1, :, :-1] #-1 in 3rd dimension to eliminate counter from targets
    
#    assert data.shape == targets.shape #works without counter 

    labels = np.ones((targets.shape[0], targets.shape[1], 2), dtype=np.int32)
    #ensure maximum 1 melody and harmony 
    assert np.all(np.sum(targets[:, :, :preprocess.Melody_Range], axis=2) <= 1)
    assert np.all(np.sum(targets[:, :, preprocess.Melody_Range:], axis=2) <= 1)
    #create melody and harmony labels
    labels[:, :, 0] = np.argmax(targets[:, :, :preprocess.Melody_Range], axis=2)
    labels[:, :, 1] = np.argmax(targets[:, :, preprocess.Melody_Range:], axis=2)
    targets = labels
    
    # ensure data and target integrity 
    assert targets.shape[:2] == data.shape[:2]
    assert data.shape[0] == num_time_steps * time_batch_len 

    # split sequences into time batches
    tb_data = np.split(data, num_time_steps, axis=0)
    tb_targets = np.split(targets, num_time_steps, axis=0)

    return (tb_data, tb_targets)

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

###############################################################################
    
    
np.random.seed(0) 
softmax = True     

time_batch_len = 128
loc = preprocess.pickle_loc
counter = preprocess.counter
model_dir = 'models'
run_name = time.strftime("%m%d_%H%M")


resolution = 480
time_step = 120
model_class = NottinghamModel
with open(loc, 'r') as f:
    pickle = cPickle.load(f)
    chord_to_idx = pickle['chord_to_idx']

input_dim = pickle["train"][0].shape[1]  #+1 for counter
print 'Finished loading data, input dim: {}'.format(input_dim)

#Start Batching 

data_split = {}
for dataset in ['train', 'valid']:
    
    # For testing, use ALL the sequences in base model
    # for counter, use same length of testing and training 
    if counter == False:         
        if dataset == 'valid':
            max_time_batches = -1
    else:
        max_time_batches = 10

    sequences = pickle[dataset]
    metadata = pickle[dataset + '_metadata']
    dims = sequences[0].shape[1]
    sequence_lens = [s.shape[0] for s in sequences]

    avg_seq_len = sum(sequence_lens) / len(sequences)

    print "Dataset: {}".format(dataset)
    print "Ave. Sequence Length: {}".format(avg_seq_len)
    print "Max Sequence Length: {}".format(time_batch_len)
    print "Number of sequences: {}".format(len(sequences))
    print "____________________________________"    

    batches = defaultdict(list)
#
   # for zero padding, comment out for truncating sequences
    for sequence in sequences:
        if (sequence.shape[0]-1) % time_batch_len == 0  :
            num_time_steps = ((sequence.shape[0]-1) // time_batch_len) 
        else:
            #calculate the pad size and create new sequence
            num_time_steps = ((sequence.shape[0]-1) // time_batch_len) + 1
            pad = np.zeros((num_time_steps*time_batch_len+1, sequence.shape[1]))
            pad[:sequence.shape[0],:sequence.shape[1]] = sequence
            sequence = pad

        if num_time_steps < 1:
            continue
        if max_time_batches > 0 and num_time_steps > max_time_batches:
            continue
        
#        #for truncating sequences, comment out with zero padding and counter 
#        for sequence in sequences:
#            # -1 because we can't predict the first step
#            num_time_steps = ((sequence.shape[0]-1) // time_batch_len) 
#            if num_time_steps < 1:
#                continue
#            if max_time_batches > 0 and num_time_steps > max_time_batches:
#                continue
        batches[num_time_steps].append(sequence)

        # create batches of examples based on sequence length/minibatches
        dataset_data =  [batch_data(b, n) for n, b in batches.iteritems()]
        
        #add metadata to batched data (just in case)
        data_split[dataset] = {
        "data": dataset_data,
        "metadata": metadata,
        }
        data_split["input_dim"] = dataset_data[0][0][0].shape[2]


# set up run dir
run_folder = os.path.join(model_dir, run_name)
if os.path.exists(run_folder):
    raise Exception("Run name {} already exists, choose a different one", format(run_folder))
os.makedirs(run_folder)

#start logger for training and validation runs
logger = logging.getLogger(__name__) 
logger.handlers = []
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(os.path.join(run_folder, "training.log")))


#setup a grid for trying combinations of hyperparameters
grid_search = {
    "dropout_prob": [0.8],
    "input_dropout_prob": [0.9],
    "melody_coeff": [0.5],
    "num_layers": [1,2],
    "hidden_size": [50],
    "num_epochs": [3],
    "learning_rate": [5e-3],
    "learning_rate_decay": [0.9],
}

# Generate product of hyperparams
runs = list(list(itertools.izip(grid_search, x)) for x in itertools.product(*grid_search.itervalues()))
logger.info("{} runs detected".format(len(runs)))
#
best_config = None
best_valid_loss = None
time_batch_len=128
comb = 1

for combination in runs:
    #load grid values to config
    config = DefaultConfig()
    config.dataset = 'softmax'
    #create model with random name
    config.model_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(12)) + '.model'
    for attr, value in combination:
        setattr(config, attr, value)

    config.input_dim = data_split["input_dim"]
    
    print ''
    print'Combination no. {}'.format(comb)
    print ''
    logger.info(config)
    config_file_path = os.path.join(run_folder, get_config_name(config) + 'RUN_'+str(comb)+'_'+ '.config')
    with open(config_file_path, 'wb') as f: 
        cPickle.dump(config, f)
#
#%%
    # build tensorflow models for training and validation from model.py
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            train_model = model_class(config, training=True)
        with tf.variable_scope("model", reuse=True):
            valid_model = model_class(config, training=False)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=40)
        tf.initialize_all_variables().run()
        
        # training
        early_stop_best_loss = None
        start_saving = False
        saved_flag = False
        train_losses, valid_losses = [], []
        start_time = time.time()

        for i in range(config.num_epochs):
            loss = run_epoch(session, train_model, 
                data_split["train"]["data"], training=True, testing=False)
            train_losses.append((i, loss))
            if i == 0:
                continue

            valid_loss = run_epoch(session, valid_model, data_split["valid"]["data"], training=False, testing=False)
            valid_losses.append((i, valid_loss))
            logger.info('Epoch: {}, Train Loss: {}, Valid Loss: {}, Time Per Epoch: {}'.format(\
                    i, loss, valid_loss, (time.time() - start_time)/i))
            
            # save current model if new validation loss goes higher or lower than current best validation loss
            if early_stop_best_loss == None:
                early_stop_best_loss = valid_loss
            elif valid_loss < early_stop_best_loss:
                early_stop_best_loss = valid_loss
                if start_saving:
                    logger.info('Best loss so far encountered, saving model.')
                    saver.save(session, os.path.join(run_folder, config.model_name))
                    saved_flag = True
            elif not start_saving:
                start_saving = True 
                logger.info('Valid loss increased for the first time, will start saving models')
                saver.save(session, os.path.join(run_folder, config.model_name))
                saved_flag = True
                
        #save model if not saved already
        if not saved_flag:
            saver.save(session, os.path.join(run_folder, config.model_name))
#
        #plot train and validation loss curves
        axes = plt.gca()
        axes.set_ylim([0, 3])

        plt.plot([t[0] for t in train_losses], [t[1] for t in train_losses])
        plt.plot([t[0] for t in valid_losses], [t[1] for t in valid_losses])
        plt.legend(['Train Loss', 'Validation Loss'])
        chart_file_path = os.path.join(run_folder, get_config_name(config) +'_RUN_'+str(comb)+'_'+'.png')
        plt.savefig(chart_file_path)
        plt.clf()

        #log the best model and config   
        logger.info("Config {}, Loss: {}".format(config, early_stop_best_loss))
        if best_valid_loss == None or early_stop_best_loss < best_valid_loss:
            logger.info("Found best new model!")
            best_valid_loss = early_stop_best_loss
            best_config = config
        
        comb = comb+1

    logger.info("Best Config: {}, Loss: {}".format(best_config, best_valid_loss))
