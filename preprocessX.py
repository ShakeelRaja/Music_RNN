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
import copy 
from pprint import pprint
import itertools
import midi_util
import mingus
import mingus.core.chords
import matplotlib.pyplot as plt

# combine sequences with counters based on set flags 
def attach(data, counter):
    result = []
    result = [np.hstack((a[0], np.array(b[0])))]
    for i in range(1, len(a)):
        result.append(np.hstack((a[i], np.array(b[i]))))
    final[type] = result
    return None 

# function to reduce all possible chords to either major and minor of each key 
def Resolve_Chords(chord):
    # to eliminate chords played with two keys 
    chordEliminate = ['major third', 'minor third', 'perfect fifth']
    if chord in chordEliminate:
        return None
    
    # take the first chord if two chords exist 
    if "|" in chord:
        chord = chord.split("|")[0]
    
    # remove 7 , 11, 9 and 6th chord and replace with relevant major or minor
    if chord.endswith("7") or chord.endswith("9") or chord.endswith("6"):
        chord = chord[:-1]
    if chord.endswith("11"):
        chord = chord[:-2] 
    
    # replace diminished chord with minor
    if chord.endswith("dim"):
        chord = chord[:-3] + "m"
    
    # add M to all Major chords
    if (not chord.endswith("m") and not chord.endswith("M")) or chord.endswith('#'):
       chord = chord +'M'
    
    return chord
   

def Data_to_Sequence(midFile, timeStep):
    unkChord = 'NONE'   # for unknown chords
    harmony = []    
    # get MIDI evet messages from each file 
    midData =  midi.read_midifile(midFile)
    
    # store file path and name as meta info
    meta = {
        "path": midFile,
        "name": midFile.split("/")[-1].split(".")[0]
    }
    # check for length , 3 tracks for meta , melody and harmony
    if len(midData) != 3:
        return (meta, None)
    
    for msg in midData[0]:
        
        if isinstance(msg, midi.TimeSignatureEvent):
            # get PPQN (Pulse per Quarter Note)for MIDI resolution 
            meta["ticks_per_quarter_note"] = msg.get_metronome()
            
            # get time signature values 
            num = midData[0][2].data[0]
            dem  = 2** (midData[0][2].data[1])
            sig = (num, dem)
            meta["signature"] = sig
            
            # Measure time signatur frequency
            if sig not in sigs:
                sigs[sig] = 1
            else:
                sigs[sig] += 1
            
            # Filter out sequences with time signature 4/4 based on flag 
            if fourByFour == True:
                if (num == 3 or num == 6) or (dem !=4):
                    return (meta, None)

    # Track ingestion   
    nTicks = 0
    
    # get melody and harmony notes and ticks from midi Data
    melNotes, melTicks = midi_util.ingest_notes(midData[1])
    harNotes, harTicks = midi_util.ingest_notes(midData[2])
    
    # round number of ticks with given time step value
    nTicks = midi_util.round_tick(max(melTicks, harTicks), timeStep)
    
    # get melody encodings mapped to defined melody range 
    melSequence = midi_util.round_notes(melNotes, nTicks, timeStep, R=melodyRange, O=melodyMin)

    # filter out sequences with a double note is found in melody
    for i in range(melSequence.shape[0]):
        if np.count_nonzero(melSequence[i, :]) > 1:
            return (meta, None)
    
    # get harmony sequence and process with Mingus
    harSequence = midi_util.round_notes(harNotes, nTicks, timeStep)
    
    # convert sharps to flats to consider compositional considerations
    flat_note = {"A#": "Bb", "B#": "C", "C#": "Db", "D#": "Eb", "E#": "F", "F#": "Gb", "G#": "Ab",}

    # use Mingus to identify chords from harmony notes
    for i in range(harSequence.shape[0]):
        
        # get note data 
        notes = np.where(harSequence[i] == 1)[0]
        if len(notes) > 0:
            
            # get note names without octave information
            noteName = [ mingus.core.notes.int_to_note(note%12) for note in notes]
            chordName = mingus.core.chords.determine(noteName, shorthand=True)
            
            if len(chordName) == 0:
                # convert to flat if chord not identified and try again
                noteName = [ flat_note[n] if n in flat_note else n for n in noteName]
                chordName = mingus.core.chords.determine(noteName, shorthand=True)
            
            if len(chordName) == 0:
                # if chord does not exist, label as NONE
                if len(harmony) > 0:
                    harmony.append(harmony[-1])
                else:
                    harmony.append(unkChord)
                    
            # resolve chords as major or minor for other types of chord
            else:
                resolvedChord = Resolve_Chords(chordName[0])
                if resolvedChord:
                    harmony.append(resolvedChord)
                else:
                    harmony.append(unkChord)
        else:
            # label as unresolve after all attempts
            harmony.append(unkChord)
            
    return (meta, (melSequence, harmony))

def Parse_Data(directory, timeStep):
    #returns a list of [T x D] matrices, where each matrix represents a 
    #a sequence with T time steps over D dimensions
    midFiles = [ os.path.join(directory, d) for d in os.listdir(directory)
              if os.path.isfile(os.path.join(directory, d))] 
    
    # retrieve melody and harmony sequences from files  
    midiSequences = [Data_to_Sequence(f, timeStep=timeStep) for f in midFiles ]
    
    # filter out the sequence if 2 tracks of melody and harmony are not found
    midiSequences = filter(lambda x: x[1] != None, midiSequences)

    return midiSequences


def combine(mel, har):
    
    unkChord = 'NONE' # for unknown chords
    final = np.zeros((mel.shape[0], melodyRange + numChords))
   
    # for all melody sequences that don't have any notes, add the empty melody marker (last one)
    for i in range(mel.shape[0]):
        if np.count_nonzero(mel[i, :]) == 0:
            mel[i, melodyRange -1] = 1
    # add all the melodies
    final[:, :mel.shape[1]] += mel
    
    # store chords on NONE if not found
    harIdx = [ chordMap[x] if x in chordMap else chordMap[unkChord] for x in har ]
    harIdx = [ melodyRange + y for y in harIdx ]
    
    # return combined melody and harony one hot vector enconding
    # final vector will have exactly two 1's ( for melody and hamrmony)
    final[np.arange(len(har)), harIdx] = 1
   
    return final

# array shifting for appending counterswith shifted values
def shift(l,n):
    return itertools.islice(itertools.cycle(l),n,n+len(l))
#%%############################################################################
"""
PreProcessing for structure augmented LSTMs

The program uses following flags to create different datasets depending
on the experiment. Setting a certain flag to True would enable the code
to augment the selected features into the dataset it creates after pre-processing

"""
zeroPad = True
counter = True
normCount = True
Shift = True

counterQB = False
counterB = True
fourByFour = False

# Define original data and Saved data location

# For final experiment

dataLoc = 'data/Nottingham/{}'
pickleLoc= 'data/nottingham_allin_notStep.pickle'


#For experimentation, comment for actual experiments
#pickleLoc = 'data/nottingham_subset.pickle'
#data_loc = 'data/Nottingham_subset/{}'

"""
Program variables definition and initialization 
"""

# Define Range of melody to be used based on MIDI note numbers

melodyMax = 88      # E6
melodyMin = 55      # G3
melodyRange = melodyMax - melodyMin + 1 + 1 # +1 for rest 

# Highest midi note used for detecting harmonies

chordLimit=64       # E4

# Initilize arrays and dictionaries for pre-processing 

sigs = {}           # time signatures
data = {}           # storing data prior to combining
final = {}          # final dataset
chords = {}         # store chords from Mingus
sequenceLength = [] # length of sequences
sequenceMax = 0     # maximum sequence length , 0 be default
sequenceMin = 1000  # minimum sequence length
timeStep = 120      # time step chosen for parsing data files

"""
Main pre-prcessing section
"""
 
if __name__ == "__main__":
    
    # Splitting and Parsing MIDI files 
    
    for type in ["train", "valid"]:
        print "Parsing {} data".format(type)
        print "-----------------------------"
        
        # get parsed data from dataset
        parsedData = Parse_Data(dataLoc.format(type), timeStep)
        # meta information from index 0 
        meta = [m[0] for m in parsedData]
        # melodic and harmonic sequences at index 1
        seqPased = [s[1] for s in parsedData]
        data[type] = seqPased          # store harmony and melody vectors
        data[type + '_meta'] = meta    # store meta information 
       
        #store sequence lengths for analysis
        seqLengths = [len(seq[1]) for seq in seqPased]

        #print seqLengths
        print "Maximum sequence length found:  {}".format(max(seqLengths))
        print "Minimum sequence length found:  {}".format(min(seqLengths))
        print ""
        sequenceLength += seqLengths    # for calculating average sequence length
        sequenceMax = max(sequenceMax, max(seqLengths))
        sequenceMin = min(sequenceMin, min(seqLengths))


#   Following code will extract extra features based on the duration of song in 
#   timesteps, normalising the counter values ad shifting the counter back , based 
#   on selected flags.
        
##  COUNTER + NORMALIZATION

        # create counter vector , reverse and store in the data dictionary for appending later
        if counter == True:
            length = []
            rev = []
            shifts = []
            length += [range(x) for x in seqLengths]    # calculate song length
            for y in length:
                y = y[::-1]     # reverse the counter as "countDOWN"
                # normalize counter values 
                if normCount == True:
                    y2 = copy.deepcopy(y)
                    norm = [float(i)/max(y) for i in y] # optional normalize
                    y = norm
                tt = ([[item] for item in y])
                rev +=[tt]      # append reversed counter 
            # append counter to finl data     
            data[type + '_count'] = rev
            
## SHIFTED COUNTER

        #Calculate the value of shifts required to move 0 back to beginning of last note 
        if Shift == True:
            # deep copy reverse array for shifting 
            rev2 = copy.deepcopy(rev)
            shifted = []
            #swap current and previous until note changes
            for x, y in seqPased:
                n = -1
                now = x[n,:]
                prev = x[n-1,:]
                
                # step back from the end of song to find the beginning of last note
                # to find the number of shifts required
                while (now == prev).all() :
                    n = n-1
                    now = x[n,:]
                    prev = x[n-1,:]
                shifts += [abs(n+1)]

            #c roll back counter to point the zero at beginning of last note    
            for count, roll in zip(rev2,shifts):
                [count.append([0]) for x in range(roll)]                    
                count = list(shift(count, roll))
                count = count[:-roll]
                shifted += [count] 
            
            # append shifted counter values to final data 
            data[type + '_count_shift'] = shifted

## TIMESTEP COUNTER TO ONE QUARTER BEAT

        #adding quaterbeat/4 timestep information         
        if counterQB == True:
           # create the quater beat counter one hot vector
            tSteps = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
            #tSteps = [[1], [2], [3], [4]] 
            tStepFinal = []
            
            # duration , repitition of quarterbeats and remianing timesteps
            for dur in seqLengths:
                rep = dur//4    # number of quarterbeats in the sequence (for 4/4)
                rem = dur%4     # extra number of quarter beats (considering 4/4)
                tStepVec = []
                for _ in range(rep):
                    tStepVec += tSteps
                if rem > 0:
                    padStep = []
                    for _ in range(rem):
                        # Caculate length of padding of 0 for pre-song notes
                        padStep += [[0,0,0,0]]
                        #padStep += [[0]]
                    tStepFinal += [padStep+ tStepVec]
                else:
                    tStepFinal += ([tStepVec])
                    
            # append quarter beat vector to final dataset     
            data[type + '_countQB'] = tStepFinal

# TIMESTEP COUNTER TO FULL BEAT in 4/4
       
        # calculating and adding full beat one hot vector
        if counterB == True:
            # create one hot vectors 
            tSteps = [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],  
                      [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], 
                      [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],
                      [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1]]
            tStepFinal = []
            for dur in seqLengths:
                beat = dur//16
                rep = dur//4
                rem = dur%4
                # duration , repitition of quarterbeats and remianing timesteps
                #print dur, rep, rem
                tStepVec = []
                x = rem + 1
                y = 0
                # cycle through tSteps and calculate the relevant one hot vectora
                # at each timestep 
                while x <= dur:
                    tStepVec += [tSteps[y]]
                    y = 0 if y == 15 else y+1
                    x = x+1
                # add extra padding 
                if rem > 0:
                    padStep = []
                    for _ in range(rem):
                        padStep += [[0,0,0,0]]
                    tStepFinal += [padStep+ tStepVec]
                else:
                    tStepFinal += ([tStepVec])
                    
            # append full beat vector to final dataset
            data[type + '_countB'] = tStepFinal
        
        # count frequencies of chord occurances in the dataset
        for _, harmonySeq in seqPased:
            for c in harmonySeq:
                if c not in chords:
                    chords[c] = 1
                else:
                    chords[c] += 1        
                    
    #Calculate average length, which may be used for identifying batch timestep length 
    avg_seq = float(sum(sequenceLength)) / len(sequenceLength)
        
    #Prepare chord index for harmony one hot vector    
    chordLimit=64
    # calculate chords and frequences 
    chords = { chord: ind for chord, ind in chords.iteritems() if chords[chord] >= chordLimit }
    chordMap = { chord: ind for ind, chord in enumerate(chords.keys()) }
    numChords = len(chordMap)
    
    # append chord index to final 
    final['chordIx'] = chordMap
    
    #plot the chord distribution chart 
    pprint(chords)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(chords)), chords.values())
    plt.xticks(range(len(chords)), chords.keys())
    plt.show()
    
    #print sequence information 
    print "Total sequences parsed:      {}".format(len(sequenceLength))
    print "Maximum length of sequences: {}".format(sequenceMax)
    print "Minimum length of sequeqnces:{}".format(sequenceMin)
    print "Average length of sequences: {}".format(avg_seq)
    print "Number of chords found:      {}".format(numChords)

#%%


    #Combine melody and harmony vectors      
    for item in ["train", "valid"]:
        print "Combining {}".format(item)
        #combine melody and hamorny one hot vectors into a single vector           
        final[item] = [ combine(mel, har) for mel, har in data[item] ]
        final[item + '_meta'] = data[item + '_meta']
        
        #save pickle data with optional counters
        if counter == True:        
            a = final[item]
            if Shift == True:
                b = data[item + '_count_shift']
            else:
                b = data[item + '_count']
            attach(a,b)
        if counterQB == True:
            a = final[item]
            b = data[item + '_countQB']
            attach(a,b)
        if counterB == True:
            a = final[item]
            b = data[item + '_countB']
            attach(a,b)
        
        # save final dataset to pickle location 
        filename=pickleLoc
        with open(filename, 'w') as f:
            cPickle.dump(final, f, protocol=-1)
