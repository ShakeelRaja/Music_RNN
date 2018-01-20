from __future__ import print_function
import numpy as np
import tensorflow as tf    
from tensorflow.contrib import rnn

#import tensorflow.contrib.rnn.RNNCell as rnn_cell

class Model(object):
    """ 
    Cross-Entropy Naive Formulation
    A single time step may have multiple notes active, so a sigmoid cross entropy loss
    is used to match targets.

    seq_input: a [ T x B x D ] matrix, where T is the time steps in the batch, B is the
               batch size, and D is the amount of dimensions
    """
    
    def __init__(self, config, training=False):
        
        def create_cell(input_size):
            if cell_type == "Vanilla":
                cell_class = rnn.BasicRNNCell
            elif cell_type == "GRU":
                cell_class = rnn.GRUCell
            elif cell_type == "LSTM":
                cell_class = rnn.BasicLSTMCell
            else:
                raise Exception("Invalid cell type: {}".format(cell_type))
            cell = cell_class(hidden_size, forget_bias=1.0)
            DROP 
#            cell = cell_class(hidden_size, input_size = input_size)

            #apply output dropout to training data  
#            if training:
#                return cell.DropoutWrapper(cell, output_keep_prob = dropout_prob)
#            else:
            return cell
            
            
            
        self.config = config
        self.time_batch_len = time_batch_len = config.time_batch_len
        self.input_dim = input_dim = config.input_dim
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        dropout_prob = config.dropout_prob
        input_dropout_prob = config.input_dropout_prob
        cell_type = config.cell_type
        
        self.Melody_Max = 88
        self.Melody_Min = 55
        self.Melody_Range = self.Melody_Max - self.Melody_Min + 1 + 1

        self.seq_input = tf.placeholder(tf.float32, shape=[None, self.time_batch_len, input_dim])
        
        if training:
            self.seq_input_dropout = tf.nn.dropout(self.seq_input, keep_prob = input_dropout_prob)
        else:
            self.seq_input_dropout = self.seq_input

        self.output_dim= self.input_dim -1

        #output_dim= self.input_dim -1 # for counter reduction in the output    
        # setup variables
        with tf.variable_scope("rnn"):
            output_W = tf.get_variable("output_w", [hidden_size, self.output_dim])
            output_b = tf.get_variable("output_b", [self.output_dim])
            self.lr = tf.constant(config.learning_rate, name="learning_rate")
            self.lr_decay = tf.constant(config.learning_rate_decay, name="learning_rate_decay")

 #       cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)


        #create input sequence applying dropout to training data


        # create an n layer (num_layer) sized MultiRnnCell, defining sizes for each
        self.cell = rnn.MultiRNNCell(
            [create_cell(input_dim)] + [create_cell(hidden_size) for i in range(1, num_layers)])

        # batch size = number of timesteps i.e. 128 , initial 0 state and input+dropout tensor
        batch_size = tf.shape(self.seq_input_dropout)[0]
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        inputs_list = tf.unstack(self.seq_input_dropout, time_batch_len, 1)

        # rnn outputs a list of [batch_size x H] outputs
        outputs_list, self.final_state = rnn.static_rnn(self.cell, inputs_list, initial_state=self.initial_state)
        
        # get the outputs, calculate output activations
        outputs = tf.stack(outputs_list)
        outputs_concat = tf.reshape(outputs, [-1, hidden_size])
        logits_concat = tf.matmul(outputs_concat, output_W) + output_b

        #Reshape output tensor
        logits = tf.reshape(logits_concat, [-1, self.time_batch_len, self.output_dim]) 

        # probabilities of each note
        self.probs = self.calculate_probs(logits)
        self.loss = self.init_loss(logits, logits_concat)
        self.train_step = tf.train.RMSPropOptimizer(self.lr, decay = self.lr_decay) \
                            .minimize(self.loss)

    def init_loss(self, outputs, _):
        self.seq_targets = \
            tf.placeholder(tf.float32, [None, self.time_batch_len,  self.input_dim])

        batch_size = tf.shape(self.seq_input_dropout)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(outputs, self.seq_targets)
        return tf.reduce_sum(cross_ent) / self.time_batch_len / tf.to_float(batch_size)

    def calculate_probs(self, logits):
        return tf.sigmoid(logits)

    def get_cell_zero_state(self, session, batch_size):
#        return tf.get_default_session().run(self.cell.zero_state(batch_size, tf.float32)).eval(session=session)
#        self.cell.zero_state = tf.convert_to_tensor(self.cell.zero_state)
        
        return self.cell.zero_state(batch_size, tf.float32)

class NottinghamModel(Model):
    """ 
    Dual softmax formulation 

    A single time step should be a concatenation of two one-hot-encoding binary vectors.
    Loss function is a sum of two softmax loss functions over [:r] and [r:] respectively,
    where r is the number of melody classes
    """

    def init_loss(self, outputs, outputs_concat):
        self.seq_targets = \
            tf.placeholder(tf.int64, [ None, self.time_batch_len, 2])
        batch_size = tf.shape(self.seq_targets)[0]

        with tf.variable_scope("rnn"):
            self.melody_coeff = tf.constant(self.config.melody_coeff)

        r = self.Melody_Range
        targets_concat = tf.reshape(self.seq_targets, [-1, 2])

        melody_loss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits = outputs_concat[:, :r], \
            labels = targets_concat[:, 0])
        harmony_loss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits = outputs_concat[:, r:], \
            labels = targets_concat[:, 1])
        losses = tf.add(self.melody_coeff * melody_loss, (1 - self.melody_coeff) * harmony_loss)
        return tf.reduce_sum(losses) / self.time_batch_len / tf.to_float(batch_size)

    def calculate_probs(self, logits):
        steps = []
        for t in range(self.time_batch_len):
            melody_softmax = tf.nn.softmax(logits[:, t, :self.Melody_Range])
            harmony_softmax = tf.nn.softmax(logits[:, t, self.Melody_Range:])
            steps.append(tf.concat([melody_softmax, harmony_softmax], 1))
        return tf.stack(steps)

    def assign_melody_coeff(self, session, melody_coeff):
        if melody_coeff < 0.0 or melody_coeff > 1.0:
            raise Exception("Invalid melody coeffecient")

        session.run(tf.assign(self.melody_coeff, melody_coeff))

