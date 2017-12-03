#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:24:52 2017

@author: shaks
"""



import numpy as np
import os
import midi
import cPickle
from pprint import pprint

import midi_util
import mingus
import mingus.core.chords
import itertools

from model import Model, NottinghamModel
import tensorflow as tf

import sys
import logging
from random import shuffle
import time
import random
import string
from collections import defaultdict
import midi
import matplotlib.pyplot as plt

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
    targets = targets[:-1, :, :-1] #-1 third axis to remove counter from targets 
    
    if counter == False:
        assert data.shape == targets.shape #works without counter 

    labels = np.ones((targets.shape[0], targets.shape[1], 2), dtype=np.int32)

    #number of melody and harmony vectors must not contain more than one '1'
    assert np.all(np.sum(targets[:, :, :Melody_Range], axis=2) <= 1)
    assert np.all(np.sum(targets[:, :, Melody_Range:], axis=2) <= 1)

    #set melody and harmony targets along the third axis
    labels[:, :, 0] = np.argmax(targets[:, :, :Melody_Range], axis=2)
    labels[:, :, 1] = np.argmax(targets[:, :, Melody_Range:], axis=2)
    targets = labels
    
    # ensure data and target integrity 
    assert targets.shape[:2] == data.shape[:2]
    assert data.shape[0] == num_time_steps * time_batch_len 

    # split sequences into time batches
    tb_data = np.split(data, num_time_steps, axis=0)
    tb_targets = np.split(targets, num_time_steps, axis=0)

    assert len(tb_data) == len(tb_targets) == num_time_steps
    for i in range(len(tb_data)):
        assert tb_data[i].shape[0] == time_batch_len
        assert tb_targets[i].shape[0] == time_batch_len
#        if softmax:
#            assert np.all(np.sum(tb_data[i], axis=2) == 2)
    return (tb_data, tb_targets)


def resolve_chord(chord):
    #change 7, 9, 11 and other un-resolved chords
    if chord in CHORD_BLACKLIST:
        return None
    # take the first of dual chords
    if "|" in chord:
        chord = chord.split("|")[0]
    # remove 7ths, 11ths, 9s, 6th,
    if chord.endswith("11"):
        chord = chord[:-2] 
    if chord.endswith("7") or chord.endswith("9") or chord.endswith("6"):
        chord = chord[:-1]
    # replace 'dim' with minor
    if chord.endswith("dim"):
        chord = chord[:-3] + "m"
    if (not chord.endswith("m") and not chord.endswith("M")) or chord.endswith('#'):
       chord = chord+'M'
    return chord
   
    
def Parse_Data(input_dir, time_step, verbose=False):
    #returns a list of [T x D] matrices, where each matrix represents a 
    #a sequence with T time steps over D dimensions
    files = [ os.path.join(input_dir, f) for f in os.listdir(input_dir)
              if os.path.isfile(os.path.join(input_dir, f)) ] 
    sequences = [ \
        Data_to_Sequence(f, time_step=time_step, verbose=verbose) \
        for f in files ]
    if verbose:
        print "Total sequences: {}".format(len(sequences))
    # filter out the non 2-track MIDI's
    sequences = filter(lambda x: x[1] != None, sequences)
    if verbose:
        print "Total sequences left: {}".format(len(sequences))

    return sequences

def Data_to_Sequence(input_filename, time_step, verbose=False):

    pattern = midi.read_midifile(input_filename)
    metadata = {
        "path": input_filename,
        "name": input_filename.split("/")[-1].split(".")[0]
    }
    if len(pattern) != 3:
        if verbose:
            "Skipping track with {} tracks".format(len(pattern))
        return (metadata, None)
    # ticks_per_quarter = -1
    for msg in pattern[0]:
        
        if isinstance(msg, midi.TimeSignatureEvent):
            metadata["ticks_per_quarter"] = msg.get_metronome()
            num = pattern[0][2].data[0]
            dem  = 2** (pattern[0][2].data[1])
            sig = (num, dem)
            metadata["signature"] = sig

    # Track ingestion stage
    track_ticks = 0

    melody_notes, melody_ticks = midi_util.ingest_notes(pattern[1])
    harmony_notes, harmony_ticks = midi_util.ingest_notes(pattern[2])

    track_ticks = midi_util.round_tick(max(melody_ticks, harmony_ticks), time_step)
    if verbose:
        print "Track ticks (rounded): {} ({} time steps)".format(track_ticks, track_ticks/time_step)
    
    melody_sequence = midi_util.round_notes(melody_notes, track_ticks, time_step, 
                                  R=Melody_Range, O=Melody_Min)

    for i in range(melody_sequence.shape[0]):
        if np.count_nonzero(melody_sequence[i, :]) > 1:
            if verbose:
                print "Double note found: {}: {} ({})".format(i, np.nonzero(melody_sequence[i, :]), input_filename)
            return (metadata, None)
        
    harmony_sequence = midi_util.round_notes(harmony_notes, track_ticks, time_step)

    harmonies = []
    SHARPS_TO_FLATS = {
    "A#": "Bb",
    "B#": "C",
    "C#": "Db",
    "D#": "Eb",
    "E#": "F",
    "F#": "Gb",
    "G#": "Ab",
    }
    #Identify chords from track 1 notes using mingus library 
    for i in range(harmony_sequence.shape[0]):
        notes = np.where(harmony_sequence[i] == 1)[0]
        if len(notes) > 0:
            notes_shift = [ mingus.core.notes.int_to_note(h%12) for h in notes]
            chord = mingus.core.chords.determine(notes_shift, shorthand=True)
            if len(chord) == 0:
                # try flat combinations
                notes_shift = [ SHARPS_TO_FLATS[n] if n in SHARPS_TO_FLATS else n for n in notes_shift]
                chord = mingus.core.chords.determine(notes_shift, shorthand=True)
            if len(chord) == 0:
                #print "Could not determine chord: {} ({}, {}), defaulting to last steps chord" \
                #          .format(notes_shift, input_filename, i)
                if len(harmonies) > 0:
                    harmonies.append(harmonies[-1])
                else:
                    harmonies.append(NO_CHORD)
            else:
                resolved = resolve_chord(chord[0])
                if resolved:
                    harmonies.append(resolved)
                else:
                    harmonies.append(NO_CHORD)
        else:
            harmonies.append(NO_CHORD)
    return (metadata, (melody_sequence, harmonies))

def combine(melody, harmony):
    full = np.zeros((melody.shape[0], Melody_Range + num_chords))
    assert melody.shape[0] == len(harmony)
    # for all melody sequences that don't have any notes, add the empty melody marker (last one)
    for i in range(melody.shape[0]):
        if np.count_nonzero(melody[i, :]) == 0:
            melody[i, Melody_Range -1] = 1
    # all melody encodings should now have exactly one 1
    for i in range(melody.shape[0]):
        assert np.count_nonzero(melody[i, :]) == 1
    # add all the melodies
    full[:, :melody.shape[1]] += melody
    harmony_idxs = [ chord_mapping[h] if h in chord_mapping else chord_mapping[NO_CHORD] \
                     for h in harmony ]
    harmony_idxs = [ Melody_Range + h for h in harmony_idxs ]
    full[np.arange(len(harmony)), harmony_idxs] = 1
    # all full encodings should have exactly two 1's
    for i in range(full.shape[0]):
        assert np.count_nonzero(full[i, :]) == 2

    return full


#%%##############################################################################################
counter = True

pickle_loc = 'data/nottingham_subset.pickle'
Melody_Max = 88
Melody_Min = 55
# add one to the range for silence in melody
Melody_Range = Melody_Max - Melody_Min + 1 + 1
CHORD_BASE = 48
CHORD_BLACKLIST = [ 'major third', 'minor third', 'perfect fifth']
NO_CHORD = 'NONE'


data = {} 
store = {} 
chords = {} 
seq_lens = []
max_seq = 0
resolution = 480
time_step = 120
chord_cutoff=64
 
if __name__ == "__main__":

    # Parse midi data 
    for d in ["train", "valid"]:
        print "Parsing {}...".format(d)
        parsed = Parse_Data("data/Nottingham_subset/{}".format(d), time_step, verbose=False)
        metadata = [s[0] for s in parsed]
        seqs = [s[1] for s in parsed]
        data[d] = seqs
        data[d + '_metadata'] = metadata
        lens = [len(s[1]) for s in seqs]
        #print lens
        print "Maximum length: {}".format(max(lens))
        print ""
        seq_lens += lens
        max_seq = max(max_seq, max(lens))
        
        #create counter vector , reverse and store in the data dictionary for appending later
        if counter == True:
            l = []
            for x in lens:
                l += [range(x)]
            rev = []
            for y in l:
                y = y[::-1]
                tt = ([[item] for item in y])
                rev +=[tt]
            data[d + '_count'] = rev       
    
        # count chord frequencies from the dataset
        for _, harmony in seqs:
            for h in harmony:
                if h not in chords:
                    chords[h] = 1
                else:
                    chords[h] += 1
                    
    #Calculate average length, which may be used for identifying batch timestep length 
    avg_seq = float(sum(seq_lens)) / len(seq_lens)
        
    #Prepare chord index for harmony one hot vector    
    chords = { c: i for c, i in chords.iteritems() if chords[c] >= chord_cutoff }
    chord_mapping = { c: i for i, c in enumerate(chords.keys()) }
    num_chords = len(chord_mapping)
    store['chord_to_idx'] = chord_mapping
    
    #plot the chord distribution chart 
    pprint(chords)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(chords)), chords.values())
    plt.xticks(range(len(chords)), chords.keys())
    plt.show()
    
    #print sequence information 
    print "Number of chords: {}".format(num_chords)
    print "Max Sequence length: {}".format(max_seq)
    print "Avg Sequence length: {}".format(avg_seq)
    print "Num Sequences: {}".format(len(seq_lens))

#%%
    #Combine melody and harmony vectors        
    for d in ["train", "valid"]:
        print "Combining {}".format(d)
        #combine melody and hamorny one hot vectors into a single vector           
        store[d] = [ combine(m, h ) for m, h in data[d] ]
        store[d + '_metadata'] = data[d + '_metadata']
        
        #save pickle data with optional counter
        if counter == True:        
            a = store[d]
            b = data[d+'_count']
            result = []
            result = [np.hstack((a[0], np.array(b[0])))]
            for i in range(1, len(a)):
                result.append(np.hstack((a[i], np.array(b[i]))))
            store[d] = result
            filename= pickle_loc + "_counter"
            with open(filename, 'w') as f:
                cPickle.dump(store, f, protocol=-1)
        else:
            filename=pickle_loc
            with open(filename, 'w') as f:
                cPickle.dump(store, f, protocol=-1)
                

    
    
        
      
