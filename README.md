# Music_RNN
structure augmented RNNs for Music Generation

## REQUIREMENTS:
### Python 2.7
### TensorFlow 0.8 CPU 
(the code is being updated to run at TF 1.3 GPU with python3.5)
  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
### Python midi library:
  git+https://github.com/vishnubob/python-midi#egg=midi
### pip install following 
#### Mingus Notation Library: pip install mingus
#### matplotlib  : pip install matplotlob
#### numpy : pip install numpy

## FILES:
#### preprocess.py 
  parses the MIDI dataset (a subset of data is included in the data folder), optionally adds the sequence counter and saves resulting one ot vectors as pickle file
#### rnn.py 
  batches data according to given parameters and performs training and testing
#### model.py 
  contains the model class
#### rnn_sample.py 
  takes in the config file from saved models and generates new sequences
