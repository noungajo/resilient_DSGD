from couche import Couche

#herite de la classe Couche
class ActivationLayer(Couche):
	def __init__(self, activation, activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime
		self.siz = False
		
	#retourne l'entree d'activation
	def forward_propagation(self, input_data):
		self.input = input_data
		self.output = self.activation(self.input)
		return self.output
		
	#returns input_error=dE/dX for a given output_error=dE/dY
	#learning_rate is not used because there is no learnable paramaters
	def backward_propagation(self, output_error, learning_rate):
		back = None
		return self.activation_prime(self.input) * output_error, back