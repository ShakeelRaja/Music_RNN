import os
import argparse
import cPickle

import numpy as np
import tensorflow as tf    

import midi_util
import preprocess
from model import Model, NottinghamModel
from rnn2 import DefaultConfig

file = preprocess.pickle_loc
if __name__ == '__main__':
    
    np.random.seed()      
    
    parser = argparse.ArgumentParser(description='Script to generated a MIDI file sample from a trained model.')
    parser.add_argument('--config_file', type=str, default = '/home/shaks/Desktop/Music_RNN/models/older/using_QB/numLayers_2_hidSize_200_melCoef_0p5_dropOut_0p9_inpDropOut_0p9_timeBatchLen_128_cellType_LSTMRUN_1_.config')
    parser.add_argument('--sample_melody', action='store_true', default=False)
    parser.add_argument('--sample_harmony', action='store_true', default=False)
    parser.add_argument('--sample_seq', type=str, default='random',
        choices = ['random', 'chords'])
    parser.add_argument('--conditioning', type=int, default=16)
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
    
    print (config)
    
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
            sample_seq = midi_util.i_vi_iv_v(chord_to_idx, repeats, config.input_dim)
            print ('Sampling melody using a I, VI, IV, V progression')
    
        elif args.sample_seq == 'random':
            sample_index = np.random.choice(np.arange(len(pickle['valid'])))
            sample_seq = [ pickle['valid'][sample_index][i, :-2] 
                for i in range(pickle['valid'][sample_index].shape[0]) ]

        length = args.conditioning + args.sample_length -2
        
        QBcount = 1
        x = np.array(length)
        y = np.array(QBcount)
        chord = sample_seq[0]
        seq = [chord]
        chord = np.append(chord, x)
        chord = np.append(chord, y)
            
        if args.conditioning > 0:
            for i in range(1, args.conditioning):
                seq_input = np.reshape(chord, [1, 1, config.input_dim])
                feed = {
                    sampling_model.seq_input: seq_input,
                    sampling_model.initial_state: state,
                }
                state = session.run(sampling_model.final_state, feed_dict=feed)
                chord2 = sample_seq[i]
                chord3 = np.hstack((chord2, np.array(x)))
                chord4 = np.hstack((chord3, np.array(y)))
                chord = chord4
                if x == 0:
                    x = 0
                else:
                    x = x - np.array(1)
                if y == 4:
                    y = 1
                else:
                    y = y + np.array(1)
                
                #seq.append(chord)

        writer = midi_util.NottinghamMidiWriter(chord_to_idx, verbose=False)
        sampler = midi_util.NottinghamSampler(chord_to_idx, verbose=False)

        probb = []
        x = x - np.array(1)
        if QBcount == 1:
            y = 4
        else:
            y = y - np.array(1)

        for i in range(max(args.sample_length - len(seq), 0)):
            seq_input = np.reshape(chord, [1, 1, config.input_dim])
            print seq_input
            feed = {
                sampling_model.seq_input: seq_input,
                sampling_model.initial_state: state,
            }
            [probs, state] = session.run(
                [sampling_model.probs, sampling_model.final_state],
                feed_dict=feed)
            probs = np.reshape(probs, [config.input_dim-2])
            probb.append(probs)
            
            chord2 = sampler.sample_notes(probs)
            #chord2 = chord2[:-1]
            chord3 = np.hstack((chord2, np.array(x)))
            chord4 = np.hstack((chord3, np.array(y)))
            
            if x == 0:
                x = 0
            else:
                x = x - np.array(1)
            if y == 4:
                y = 1
            else:
                y = y + np.array(1)
           
            chord = chord4
            
            
            
            if config.dataset == 'softmax':
                r = preprocess.Melody_Range
                if args.sample_melody:
                    chord[r:] = 0
                    chord[r:] = sample_seq[i][r:]
                elif args.sample_harmony:
                    chord[:r] = 0
                    chord[:r] = sample_seq[i][:r]
    
            seq.append(chord2)
        
        #for i in rang
        seq = seq[:-2]
        for i in range(16):
            seq.append(seq[-1])
                    
        prob_matrix = np.array(probb)
        import matplotlib.pyplot as plt

        plt.imshow(prob_matrix.T, cmap='hot', interpolation='nearest')
        plt.show()
        
        writer.dump_sequence_to_midi(seq, "models/bestone3.mid", 
            time_step=time_step, resolution=resolution)