from couche import Couche
from initial import nbre_processus, grads,nbre_byzantin
import numpy as np
from threading import RLock,Event

#permet de savoir si le gradient est disponible
gradient_disponible = Event()
#permet de verrouiller une variable global
verrou_gradient = RLock()
#np.random.seed(0)
#herite de la classe couche
"""
    implemente les fonctions d'une couche a savoir foward et backward
    et genere les poids associes ainsi que le bias.
"""
"""
    implémente les fonctions d'une couche à savoir foward et backward
    et génère les poids associés ainsi que le bias.
"""
#hérite de la classe couche
class FClayer(Couche):
    # input_size = nombre de neurones d'entrée
    # output_size = nombre de neurones de sorti
	def __init__(self, input_size, output_size):
		self.weights = np.random.rand(input_size, output_size) - 0.5
		self.bias = np.random.rand(1, output_size) - 0.5
		self.siz = output_size == 10

    # returns output for a given input
	def forward_propagation(self, input_data):
		self.input = input_data
		self.output = np.dot(self.input, self.weights) + self.bias
		return self.output
		
	def backward_propagation2(self, gradient,back):
		global grads
		learning_rate = 0.001
		#après avoir agrégé on met à jour
		self.weights -= learning_rate*gradient
		self.bias -= learning_rate*back
		if len(grads) == nbre_processus - nbre_byzantin:
			grads = []

    #calcule dE/dW,dE/dB pour une erreur donnée dE/dY.retourner lerreur d'entrée dE/dX
	def backward_propagation(self,output_error, learning_rate):
		input_error = np.dot(output_error, self.weights.T)
		gradient = np.dot(self.input.T, output_error)
		global grads
		if self.siz:
			grads.append(gradient)
			#mettre à jour plustard
		else:
			self.weights -= learning_rate*gradient
			self.bias -= learning_rate*output_error 
        
		return input_error, output_error 
		
