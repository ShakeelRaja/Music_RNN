import os
import argparse
import cPickle

import numpy as np
import tensorflow as tf  
import copy   

import midi_util
import preprocess
from model import Model, NottinghamModel
from rnn2 import DefaultConfig


#file = 'data/nottingham_allin_new'
file = 'data/nottingham_allin_notStep.pickle'
#file = '/home/shaks/Desktop/Music_RNN/data/nottingham_allin_notStep_44.pickle'
def i_vi_iv_v(chord_to_idx, repeats, input_dim):
    r = preprocess.Melody_Range 
    input_dim = input_dim -5

    i = np.zeros(input_dim)
    i[r + chord_to_idx['CM']] = 1

    vi = np.zeros(input_dim)
    vi[r + chord_to_idx['Am']] = 1

    iv = np.zeros(input_dim)
    iv[r + chord_to_idx['FM']] = 1

    v = np.zeros(input_dim)
    v[r + chord_to_idx['GM']] = 1

    full_seq = [i] * 16 + [vi] * 16 + [iv] * 16 + [v] * 16
    full_seq = full_seq * repeats
    
    return full_seq
if __name__ == '__main__':
    
    np.random.seed()      
    
    parser = argparse.ArgumentParser(description='Script to generated a MIDI file sample from a trained model.')
#    parser.add_argument('--config_file', type=str, default = 'models/all_in_new/numLayers_2_hidSize_200_melCoef_0p5_dropOut_0p3_inpDropOut_1_timeBatchLen_128_cellType_LSTMRUN_18_.config')
#    parser.add_argument('--config_file', type=str, default = 'models/44_no_Ts/numLayers_2_hidSize_200_melCoef_0p5_dropOut_0p5_inpDropOut_1_timeBatchLen_128_cellType_LSTMRUN_5_.config')
    parser.add_argument('--config_file', type=str, default = 'models/all_no_Ts/numLayers_2_hidSize_200_melCoef_0p5_dropOut_0p3_inpDropOut_1_timeBatchLen_128_cellType_LSTMRUN_4_.config')
    parser.add_argument('--sample_melody', action='store_true', default=False)
    parser.add_argument('--sample_harmony', action='store_true', default=False)
    parser.add_argument('--sample_seq', type=str, default='random',
        choices = ['random', 'chords'])
    parser.add_argument('--conditioning', type=int, default=32)
    parser.add_argument('--sample_length', type=int, default=256)
    
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f: 
        config = cPickle.load(f)
    
    if config.dataset == 'softmax':
        config.time_batch_len = 1
        config.max_time_batches = -1
        model_class = NottinghamModel
        with open(file, 'r') as f:
            pickle = cPickle.load(f)
        chord_to_idx = pickle['chord_to_idx']
    
        time_step = 120
        resolution = 480
    
        # use time batch len of 1 so that every target is covered
    #    test_data = util.batch_data(pickle['test'], time_batch_len = 1, 
    #        max_time_batches = -1, softmax = True)
    #else:
    #    raise Exception("Other datasets not yet implemented")
    
    print (config)
#%%    
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            sampling_model = model_class(config)
    
        saver = tf.train.Saver(tf.all_variables())
        model_path = os.path.join(os.path.dirname(args.config_file), 
            config.model_name)
        saver.restore(session, model_path)
    
        state = sampling_model.get_cell_zero_state(session, 1)
        if args.sample_seq == 'chords':
            # 16 - one measure, 64 - chord progression
            repeats = args.sample_length / 64
            sample_seq = i_vi_iv_v(chord_to_idx, repeats, config.input_dim)
            print ('Sampling melody using a I, VI, IV, V progression')
    
        elif args.sample_seq == 'random':
            sample_index = np.random.choice(np.arange(len(pickle['valid'])))
            sample_seq = [ pickle['valid'][sample_index][i, :-5] 
                for i in range(pickle['valid'][sample_index].shape[0]) ]
          
        lenVec = [] 
        rev = []
        length = args.sample_length
        lenVec = [range(length)]
        for y in lenVec:
                y = y[::-1]
                y2 = copy.deepcopy(y)
                norm = [float(i)/max(y) for i in y] # optional normalize
                y = norm
                tt = ([[item] for item in y])
                rev +=[tt]
      
        ptr = 0 
        chord1 = sample_seq[0]
        seq = [chord1]
        chord = np.append(chord1, rev[0][ptr])

        
#        tSteps = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
#        tCount = 0 
#        chord = np.append(chord, tSteps[tCount])

            
        mSteps = [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],  
                  [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], 
                  [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],
                  [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1]]
        mCount = 0 
        chord = np.append(chord, mSteps[mCount])
#        print rev[0][ptr], tSteps[tCount] , mSteps[mCount]
        if args.conditioning > 0:
            ptr = ptr +1
            for i in range(1, args.conditioning):
                seq_input = np.reshape(chord, [1, 1, config.input_dim])
                feed = {
                    sampling_model.seq_input: seq_input,
                    sampling_model.initial_state: state,
                }
                state = session.run(sampling_model.final_state, feed_dict=feed)
                chord = sample_seq[i]
                seq.append(chord)
                
                if rev[0][ptr] == 0:
                    ptr = ptr
                else:
                    ptr = ptr + 1
                    
                chord = np.hstack((chord, rev[0][ptr]))
                
#                
#                if tCount <3:
#                    tCount +=1
#                elif tCount == 3:
#                    tCount = 0
#                    
#                chord = np.hstack((chord, tSteps[tCount]))    
#                
                if mCount <15:
                    mCount +=1
                elif mCount == 15:
                    mCount = 0
                    
                chord = np.hstack((chord, mSteps[mCount]))    

#                print rev[0][ptr], tSteps[tCount] , mSteps[mCount]

       
#%%                
#    
        if config.dataset == 'softmax':
            writer = midi_util.NottinghamMidiWriter(chord_to_idx, verbose=False)
            sampler = midi_util.NottinghamSampler(chord_to_idx, verbose=False)
        else:
            # writer = midi_util.MidiWriter()
            # sampler = sampling.Sampler(verbose=False)
            raise Exception("Other datasets not yet implemented")
        
        probb = []
        #ptr = ptr + 1
        for i in range(max(args.sample_length - len(seq)+ 1, 0)):
        #for i in range(max(args.sample_length - len(seq), 0)):
            seq_input = np.reshape(chord, [1, 1, config.input_dim])
            
            
            feed = {
                sampling_model.seq_input: seq_input,
                sampling_model.initial_state: state,
            }
            [probs, state] = session.run(
                [sampling_model.probs, sampling_model.final_state],
                feed_dict=feed)
            probs = np.reshape(probs, [config.input_dim-5])
            probb.append(probs)
            
            chordx = sampler.sample_notes(probs)
            #chord2 = chord2[:-1]
            chord = np.append(chordx, rev[0][ptr])
            x = rev[0][ptr]
            
            if float(x[0]) > 0:
                ptr = ptr + 1
               
#            chord = np.hstack((chord, tSteps[tCount]))  
#            if tCount <3:
#                tCount +=1
#            elif tCount == 3:
#                tCount = 0
#                
            chord = np.hstack((chord, mSteps[mCount]))  
            if mCount <15:
                mCount +=1
            elif mCount == 15:
                mCount = 0
                
            
            if config.dataset == 'softmax':
                r = preprocess.Melody_Range
                if args.sample_melody:
                    chord[r:] = 0
                    chord[r:] = sample_seq[i][r:]
                elif args.sample_harmony:
                    chord[:r] = 0
                    chord[:r] = sample_seq[i][:r]
            #print chordx

            seq.append(chordx)
#            print rev[0][ptr], tSteps[tCount] , mSteps[mCount]
            
        seq = seq[:-2]
        for i in range(16):
            
            seq.append(seq[-1])
            

        prob_matrix = np.array(probb)
        import matplotlib.pyplot as plt

        plt.imshow(prob_matrix.T,cmap = "Reds",  interpolation='nearest')
        plt.show()
        

        writer.dump_sequence_to_midi(seq, "models/zzGENERATED/all_noTS/all_noTS.mid", 
            time_step=time_step, resolution=resolution)