import numpy as np
import numpy.matmul as matmul


class RNN:

	pass
    # Assume passed data is always 1-d array (should use assert)
	def __init__(self, input_size, hidden_state_size, output_size):

		self.__input_size = input_size
		self.__hidden_state_size = hidden_state_size
		self.__output_size = output_size

		self.__init__weights()

		# Init timestep record
		self.__init__record()

	# Equation : ht = tanh(Uxt + (Wht−1 + bxh))
	# for output, we only produce the vector
	# We don't care on interpreting it
	def __init__weights(self):

		self.__init__input_matrix()
		self.__init__hidden_state_weights()
		self.__init__hidden_state_biases()
		self.__init__output_weights()
		self.__init__output_biases()

	def __init__input_matrix(self):
		self.__input_matrix = np.random.rand(self.__hidden_state_size, self.__input_size)

	def __init__hidden_state_weights(self):
		self.__hidden_matrix = np.random.rand(self.__hidden_state_size, self.__hidden_state_size)

	def __init__hidden_state_biases(self):
		self.__hidden_bias = np.random.rand(self.__hidden_state_size, 1)

	def __init__output_weights(self):
		self.__output_matrix = np.random.rand(self.__output_size, self.__hidden_state_size)

	def __init__output_biases(self):
		self.__output_bias = np.random.rand(self.__output_size, 1)

	# Track hidden state, output and input
	def __init__record(self):
		self.__hidden_state = []
		self.__output = []
		self.__input = []

	# To-do : Assert data size is equal to layer's size
	# Activation : tanh for input
	def propagate(self, data):
		self.__input.append(data)
		current_hidden_state = matmul(self.__input_matrix, data)
		current_hidden_state += matmul(self.__hidden_matrix, self.__hidden_state[-1])
		current_hidden_state += self.__hidden_bias

		# Hidden state
		current_hidden_state = np.tanh(current_hidden_state)
		self.__hidden_state.append(current_hidden_state)

		# Output
		# Let's just use tanh for activation now
		output = matmul(self.__output_matrix, current_hidden_state) + self.__output_bias
		output = np.tanh(output)

		self.__output.append(output)

		return output
