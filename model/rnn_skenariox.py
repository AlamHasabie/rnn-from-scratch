import numpy as np
from numpy import matmul


class RNN:

	pass
    # Assume passed data is always 1-d array (should use assert)
	def __init__(self, input_size, hidden_state_size, output_size, skenario, verbose=False):

		self.__input_size = input_size
		self.__hidden_state_size = hidden_state_size
		self.__output_size = output_size
		self.__skenario = skenario
		self.__verbose = verbose


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

		# Setup the first (initial timestep)
		# For long sequences, later our initial hidden state can be replaced
		self.__init__initial_hidden_state()

	def __init__input_matrix(self):
		self.__input_matrix = np.random.rand(self.__hidden_state_size, self.__input_size)

	def __init__hidden_state_weights(self):
		if (self.__skenario == 1) :
			#print("skenario 1")
			temp_hidden_weight = [ [ 0 for i in range(self.__hidden_state_size) ] for j in range(self.__hidden_state_size) ]
			#print("temp_hidden_weight")
			#print(temp_hidden_weight)
		elif (self.__skenario == 2) :
			temp_hidden_weight = [ [ 1 for i in range(self.__hidden_state_size) ] for j in range(self.__hidden_state_size) ]
		else :
			print("skenario 3")
			temp_hidden_weight = np.random.rand(self.__hidden_state_size, self.__hidden_state_size)		
		self.__hidden_matrix = temp_hidden_weight
		#self.__hidden_matrix = np.random.rand(self.__hidden_state_size, self.__hidden_state_size)

	def __init__hidden_state_biases(self):
		self.__hidden_bias = np.random.rand(self.__hidden_state_size, 1)

	def __init__output_weights(self):		
		if (self.__skenario == 1) :
			temp_output_weight = [ [ 0 for i in range(self.__hidden_state_size) ] for j in range(self.__output_size) ]
		elif (self.__skenario == 2) :
			temp_output_weight = [ [ 1 for i in range(self.__hidden_state_size) ] for j in range(self.__output_size) ]
		else :
			temp_output_weight = np.random.rand(self.__output_size, self.__hidden_state_size)
		self.__output_matrix = temp_output_weight
		#self.__output_matrix = np.random.rand(self.__output_size, self.__hidden_state_size)

	def __init__output_biases(self):
		self.__output_bias = np.random.rand(self.__output_size, 1)

	# Track hidden state, output and input
	def __init__record(self):
		self.__hidden_state = []
		self.__output = []
		self.__input = []

	# All zeros for now
	def __init__initial_hidden_state(self) :
		self.__current_hidden_state = np.zeros((self.__hidden_state_size,1))

	# To-do : Assert data size is equal to layer's size
	# Activation : tanh for input
	def forward(self, data):
		self.__input.append(data)
		next_hidden_state = matmul(self.__input_matrix, data)
		next_hidden_state += matmul(self.__hidden_matrix, self.__current_hidden_state)
		next_hidden_state += self.__hidden_bias

		# Hidden state
		next_hidden_state = np.tanh(next_hidden_state)
		self.__current_hidden_state = next_hidden_state
		self.__hidden_state.append(self.__current_hidden_state)

		# Output
		# Let's just use tanh for activation now
		output = matmul(self.__output_matrix, self.__current_hidden_state) + self.__output_bias
		output = np.tanh(output)

		self.__output.append(output)

		# Assert length is all same

		assert len(self.__output) == len(self.__input)
		assert len(self.__output) == len(self.__hidden_state)

		# Logging purpose
		if self.__verbose :
			step = len(self.__output) - 1
			print("step-{}, hidden state : \n{}".format(step + 1, self.__hidden_state[step]))
			print("step-{}, output state : \n{}".format(step + 1, self.__output[step]))

		return output
