import numpy as np




class Network:
	def __init__(self):
		self.layers = []
		self.loss = None
		self.loss_prime = None
		self.espion = None
		self.back =None
		
	#add layer to network
	def add(self, layer):
		self.layers.append(layer)
		
	#set loss to use
	def use(self, loss, loss_prime):
		self.loss = loss
		self.loss_prime = loss_prime
		
	#predict output for given input
	def predict (self, input_data):
		#sample dimension first
		samples = len(input_data)
		result = []
		
		#run network over all samples
		for i in range(samples):
			#foward propagation
			output = input_data[i]
			for layer in self.layers:
				output = layer.forward_propagation(output)
			result.append(output)
			
		return result
	
	def fit2(self, gradient):
		back = self.back
		self.espion.backward_propagation2(gradient,back)
		
		
	#train the network
	def fit(self, x_train, y_train, learning_rate):
			batch = len(x_train)
			batch = 1
			err = 0
			for j in range(batch):
                    #forward_propagation
				output = x_train[j]
				for layer in self.layers:
					output = layer.forward_propagation(output)

                    #compute loss(for display purpose only)
				err += self.loss(y_train[j], output)

                        #backward propagation
				error = self.loss_prime(y_train[j], output)
				for layer in reversed(self.layers):
					error, output_error = layer.backward_propagation(error, learning_rate)
					if layer.siz:
						self.espion = layer
						self.back = output_error

			return err/batch