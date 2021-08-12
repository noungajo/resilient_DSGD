#class abstraite d'une couche
class Couche:
	def __init__(self):
		self.input = None
		self.output = None
		
	#calcule de la sortie Y d'une couche pour une entrée X donnée
	def foward_propagation(self, input):
		raise NotImplementedError
		
	#calcule dE/dX pour dE/dY donné(et mettre à jour les paramètre si chaque)
	def backward_propagation(self, output_error, learning_rate):
		raise NotImplementedError