# Basic RNN module interface testing
from model.rnn import RNN
import numpy as np


# Params
sequence = 10
input_size = 20
hidden_size = 5
output_size = 10
verbose = True

rnn_layer = RNN(input_size, hidden_size, output_size, verbose)

for _ in range(sequence) :

    # Map [0,1] -> [-1,1]
    data = np.random.rand(20,1) * 2 - 1
    output = rnn_layer.forward(data)