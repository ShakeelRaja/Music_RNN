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

import matplotlib.pyplot as plt

def resolve_chord(chord):
    #change 7, 9, 11 and other un-resolved chords
    if chord in chordEliminate:
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
    # filter out the non 2-track MIDI's
    sequences = filter(lambda x: x[1] != None, sequences)

    return sequences

def Data_to_Sequence(input_filename, time_step, verbose=False):

    pattern = midi.read_midifile(input_filename)
    metadata = {
        "path": input_filename,
        "name": input_filename.split("/")[-1].split(".")[0]
    }
    if len(pattern) != 3:
        return (metadata, None)
    # ticks_per_quarter = -1
    for msg in pattern[0]:
        
        if isinstance(msg, midi.TimeSignatureEvent):
            metadata["ticks_per_quarter"] = msg.get_metronome()
            num = pattern[0][2].data[0]
            dem  = 2** (pattern[0][2].data[1])
            sig = (num, dem)
            metadata["signature"] = sig
            if sig not in sigs:
                sigs[sig] = 1
            else:
                sigs[sig] += 1
                
            if fourByFour == True:
                if (num == 3 or num == 6) or (dem !=4):

                    return (metadata, None)

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
    flat_note = {"A#": "Bb", "B#": "C", "C#": "Db", "D#": "Eb", "E#": "F", "F#": "Gb", "G#": "Ab",}
    #Identify chords from track 1 notes using mingus library 
    for i in range(harmony_sequence.shape[0]):
        notes = np.where(harmony_sequence[i] == 1)[0]
        if len(notes) > 0:
            notes_shift = [ mingus.core.notes.int_to_note(h%12) for h in notes]
            chord = mingus.core.chords.determine(notes_shift, shorthand=True)
            if len(chord) == 0:
                # try flat combinations
                notes_shift = [ flat_note[n] if n in flat_note else n for n in notes_shift]
                chord = mingus.core.chords.determine(notes_shift, shorthand=True)
            if len(chord) == 0:
                #print "Could not determine chord: {} ({}, {}), defaulting to last steps chord" \
                #          .format(notes_shift, input_filename, i)
                if len(harmonies) > 0:
                    harmonies.append(harmonies[-1])
                else:
                    harmonies.append(unkChord)
            else:
                resolved = resolve_chord(chord[0])
                if resolved:
                    harmonies.append(resolved)
                else:
                    harmonies.append(unkChord)
        else:
            harmonies.append(unkChord)
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
    harmony_idxs = [ chord_mapping[h] if h in chord_mapping else chord_mapping[unkChord] \
                     for h in harmony ]
    harmony_idxs = [ Melody_Range + h for h in harmony_idxs ]
    full[np.arange(len(harmony)), harmony_idxs] = 1
    # all full encodings should have exactly two 1's
    for i in range(full.shape[0]):
        assert np.count_nonzero(full[i, :]) == 2

    return full


#%%############################################################################
zeroPad = True
counter = True
normCount = True
Shift = True

counterQB = False
counterB = True
fourByFour = False




#data and processed data locations
pickle_loc = 'data/nottingham_allin_notStep.pickle'
data_loc = 'data/Nottingham/{}'

#pickle_loc = 'data/nottingham_subset.pickle'
#data_loc = 'data/Nottingham_subset/{}'

maxMiniBatches = 10

Melody_Max = 88
Melody_Min = 55
# add one to the range for silence in melody
Melody_Range = Melody_Max - Melody_Min + 1 + 1
#CHORD_BASE = 48
chordEliminate = [ 'major third', 'minor third', 'perfect fifth']
#unknown chord type
unkChord = 'NONE'
#chord midi value limit 
chordLimit=64

sigs = {}
data = {} 
store = {} 
chords = {} 
seq_lens = []
max_seq = 0
resolution = 480
time_step = 120

 
if __name__ == "__main__":
    # array shifting for appending counters 
    def shift(l,n):
        return itertools.islice(itertools.cycle(l),n,n+len(l))
    
    # Parse midi data 
    for d in ["train", "valid"]:
        print "Parsing Dataset : {}...".format(d)
        parsed = Parse_Data(data_loc.format(d), time_step, verbose=False)
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
        
        # count chord frequencies from the dataset
        for _, harmony in seqs:
            for h in harmony:
                if h not in chords:
                    chords[h] = 1
                else:
                    chords[h] += 1
        
        #create counter vector , reverse and store in the data dictionary for appending later
        if counter == True:
            length = []
            rev = []
            shifts = []
            length += [range(x) for x in lens]
            for y in length:
                y = y[::-1]
                if normCount == True:
                    y2 = copy.deepcopy(y)
                    norm = [float(i)/max(y) for i in y] # optional normalize
                    y = norm
                tt = ([[item] for item in y])
                rev +=[tt]
            
                
            data[d + '_count'] = rev
            
                    #Calculate the value of shifts required to move 0 back to beginning of last note 
        if Shift == True:

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

        #adding quaterbeat/4 timestep information         
        if counterQB == True:
            
            tSteps = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
            #tSteps = [[1], [2], [3], [4]] 
            tStepFinal = []
            for dur in lens:
                rep = dur//4
                rem = dur%4
                # duration , repitition of quarterbeats and remianing timesteps
                #print dur, rep, rem
                tStepVec = []
                for _ in range(rep):
                    tStepVec += tSteps
                if rem > 0:
                    padStep = []
                    for _ in range(rem):
                        padStep += [[0,0,0,0]]
                        #padStep += [[0]]
                    tStepFinal += [padStep+ tStepVec]
                else:
                    tStepFinal += ([tStepVec])
                
            data[d + '_countQB'] = tStepFinal
            
        if counterB == True:
            tSteps = [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],  
                      [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], 
                      [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],
                      [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1]]
            tStepFinal = []
            for dur in lens:
                beat = dur//16
                rep = dur//4
                rem = dur%4
                # duration , repitition of quarterbeats and remianing timesteps
                #print dur, rep, rem
                tStepVec = []
                x = rem + 1
                y = 0
                while x <= dur:
                    tStepVec += [tSteps[y]]
                    y = 0 if y == 15 else y+1
                    x = x+1
                if rem > 0:
                    padStep = []
                    for _ in range(rem):
                        padStep += [[0,0,0,0]]
                    tStepFinal += [padStep+ tStepVec]
                else:
                    tStepFinal += ([tStepVec])
            data[d + '_countB'] = tStepFinal
           
#        if counterFB == True:
#            tSteps = [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],  
#                      [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], 
#                      [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],
#                      [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1]]
#            tStepFinal = []
#            for dur in lens:
#                beat = dur//16
#                rep = dur//4
#                rem = dur%4
#                # duration , repitition of quarterbeats and remianing timesteps
#                #print dur, rep, rem
#                tStepVec = []
#                x = rem + 1
#                y = 0
#                while x <= dur:
#                    tStepVec += [tSteps[y]]
#                    y = 0 if y == 63 else y+1
#                    x = x+1
#                if rem > 0:
#                    padStep = []
#                    for _ in range(rem):
#                        padStep += [[0,0,0,0]]
#                    tStepFinal += [padStep+ tStepVec]
#                else:
#                    tStepFinal += ([tStepVec])
#            data[d + '_countFB'] = tStepFinal

                    
    #Calculate average length, which may be used for identifying batch timestep length 
    avg_seq = float(sum(seq_lens)) / len(seq_lens)
        
    #Prepare chord index for harmony one hot vector    
    chords = { c: i for c, i in chords.iteritems() if chords[c] >= chordLimit }
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
    # combine 
    def attach(data, counter):
        result = []
        result = [np.hstack((a[0], np.array(b[0])))]
        for i in range(1, len(a)):
            result.append(np.hstack((a[i], np.array(b[i]))))
        store[d] = result
        return None 

    #Combine melody and harmony vectors      
    for d in ["train", "valid"]:
        print "Combining {}".format(d)
        #combine melody and hamorny one hot vectors into a single vector           
        store[d] = [ combine(m, h ) for m, h in data[d] ]
        store[d + '_metadata'] = data[d + '_metadata']
        
        #save pickle data with optional counters
        if counter == True:        
            a = store[d]
            if Shift == True:
                b = data[d+'_count_shift']
            else:
                b = data[d+'_count']
            attach(a,b)

        if counterQB == True:
            a = store[d]
            b = data[d + '_countQB']
            attach(a,b)
        
        if counterB == True:
            a = store[d]
            b = data[d + '_countB']
            attach(a,b)
#
#        if counterFB == True:
#            a = store[d]
#            b = data[d + '_countFB']
#            attach(a,b)

        filename=pickle_loc
        with open(filename, 'w') as f:
            cPickle.dump(store, f, protocol=-1)

        
      