#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 14:43:43 2017

@author: shaks
"""

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