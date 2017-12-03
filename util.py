import os
import math
import cPickle
from collections import defaultdict
from random import shuffle

import numpy as np
import tensorflow as tf    

import preprocess

def run_epoch(session, model, batches, training=False, testing=False):
    """
    session: Tensorflow session object
    model: model object (see model.py)
    batches: data object loaded from util_data()

    training: A backpropagation iteration will be performed on the dataset
    if this flag is active

    returns average loss per time step over all batches.
    if testing flag is active: returns [ loss, probs ] where is the probability
        values for each note
    """

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
