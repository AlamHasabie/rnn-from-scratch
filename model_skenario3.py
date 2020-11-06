# Basic RNN module interface testing
from model.rnn_skenariox import RNN
import numpy as np


# Params
sequence = 5
input_size = 5
hidden_size = 5
output_size = 3
skenario = 3
verbose = True

rnn_layer = RNN(input_size, hidden_size, output_size, skenario, verbose)
print("Skenario : " + str(skenario))
for _ in range(sequence) :

    # Map [0,1] -> [-1,1]
    data = np.random.rand(input_size,1) * 2 - 1
    #print("data")
    #print(data)
    output = rnn_layer.forward(data)