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

from random import shuffle
import copy 
from collections import defaultdict
import midi
import matplotlib.pyplot as plt

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
Shift = True

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
            length = []
            rev = []
            shifts = []
            length += [range(x) for x in lens]
            for y in length:
                y = y[::-1]
                tt = ([[item] for item in y])
                rev +=[tt]
            data[d + '_count'] = rev

            #Calculate the value of shifts required to move 0 back to beginning of last note 
            if Shift == True:
                def shift(l,n):
                    return itertools.islice(itertools.cycle(l),n,n+len(l))
                rev2 = copy.deepcopy(rev)
                shifted = []
            #swap current and previous until note changes
            for x, y in seqs:
                n = -1
                now = x[n,:]
                prev = x[n-1,:]
                #print now, prev
                while (now == prev).all() :
                    n = n-1
                    now = x[n,:]
                    prev = x[n-1,:]
                shifts += [abs(n+1)]
            #calculate the chifted counter values     
            for count, roll in zip(rev2,shifts):
                [count.append([0]) for x in range(roll)]                    
                count = list(shift(count, roll))
                count = count[:-roll]
                shifted += [count] 

            data[d + '_count_shift'] = shifted
               
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
            if Shift == True:
                b = data[d+'_count_shift']
            else:
                b = data[d+'_count']
            result = []
            result = [np.hstack((a[0], np.array(b[0])))]
            for i in range(1, len(a)):
                result.append(np.hstack((a[i], np.array(b[i]))))
            store[d] = result
            filename= pickle_loc 
            with open(filename, 'w') as f:
                cPickle.dump(store, f, protocol=-1)
        else:
            filename=pickle_loc
            with open(filename, 'w') as f:
                cPickle.dump(store, f, protocol=-1)

        
      