#mnist
import numpy as np


from network import Network
from threading import Thread,RLock
from FC import FClayer
from couche_activation import ActivationLayer
from activation import tanh, tanh_prime
from perte import mse, mse_prime
from initial import nbre_byzantin,valeur_batch,epochs,mini_batch,nbre_processus,grads,byzantin

from agregation import fada

from keras.datasets import mnist
from keras.utils import np_utils

#applicage du split
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)

verrou_t = RLock()
#load mnist from server
(x_train, y_train), (x_test, y_test ) = mnist.load_data()

#training data: 60000 samples
#reshape and normalize input data

x_train = x_train.reshape(x_train.shape[0],1,28*28)
x_train = x_train.astype('float32')
x_train /= 255
x_train = x_train[:6000]
#encode output which is a number in range [0,9] into a vector of size of 10
y_train = np_utils.to_categorical(y_train)

#same for test data: 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)


		


#chargement des modèles
models = []
for i in range(nbre_processus-nbre_byzantin):
	net = Network()
	net.add(FClayer(784, 100))
	net.add(ActivationLayer(tanh, tanh_prime))
	net.add(FClayer(100,50))
	net.add(ActivationLayer(tanh, tanh_prime))
	net.add(FClayer(50,50))
	net.add(ActivationLayer(tanh, tanh_prime))
	net.add(FClayer(50,10))
	net.add(ActivationLayer(tanh, tanh_prime))
	net.use(mse, mse_prime)
	models.append(net)
	
#repartition de l'ensemble de données
t = 0
x_new = []
y_new = []
for i in range(nbre_processus-nbre_byzantin):
	pas = int(x_train.shape[0]/nbre_processus-nbre_byzantin)
	x_set = x_train[i*pas:(i+1)*pas]
	y_set = y_train[i*pas:(i+1)*pas]
	x, y = mini_batch(x_set, y_set, valeur_batch)
	x_new.append(x)
	y_new.append(y)
#sauvegarde des fichiers

fichier = open("historique_erreur/worker1.txt","w")
fichier10 = open("historique_erreur/worker10.txt","w")
fichier20 = open("historique_erreur/worker20.txt","w")
learning_rate= 0.001
#entrainement
for epoch in range(epochs):
	for batch_indice in range(x_new[0].shape[0]):
		for data in range(len(x_new)):
			for worker in range(len(models)):
			#for x, y in zip(x_set[worker],y_set[worker]):
				error = models[worker].fit(x_new[data][batch_indice], y_new[data][batch_indice], learning_rate)
				if worker==0:
					erreur1 = error
				if worker==9:
					erreur9 = error
				if worker==5:
					erreur5 = error
			gradient_byzantin = byzantin(nbre_byzantin)
			gradients = grads + gradient_byzantin
			#print("vecteur gradient", len(grads),"taille byzantin",len (gradient_byzantin),"taille total",len(gradients))
			gradient = fada(gradients)
			for worker in range(len(models)):
				models[worker].fit2(gradient)
	print("l'erreur est ",erreur1,"pour le worker 1")
		
	
	#ces workers enregistrent lhistorique de leurs erreurs à chaque epoch	
	fichier20.write(str("%0.3f"%erreur1)+",")	
	fichier10.write(str("%0.3f"%erreur9)+",")
	fichier.write(str("%0.3f"%erreur5)+",")
	
fichier.close()
fichier10.close()
fichier20.close()
	
"""
for (train_index,test_index) in kf.split(x_train):
	x_fold=x_train[train_index]
	y_fold=y_train[train_index]
	#construire le mini batch
	x_new, y_new = mini_batch(x_fold, y_fold, valeur_batch)
	#initialisation du réseau de neuronnes
	net = Network()
	net.add(FClayer(784, 100))
	net.add(ActivationLayer(tanh, tanh_prime))
	net.add(FClayer(100,50))
	net.add(ActivationLayer(tanh, tanh_prime))
	net.add(FClayer(50,50))
	net.add(ActivationLayer(tanh, tanh_prime))
	net.add(FClayer(50,10))
	net.add(ActivationLayer(tanh, tanh_prime))
	net.use(mse, mse_prime)
	learning_rate = 0.001		
	for epoch in range(epochs):
		#soumettre le mini batch
		for x,y in zip(x_new,y_new):
			#retour de l'erreur
			batch_err = net.fit(x, y, learning_rate)
		#ce worker enregistre lhistorique de ses erreurs à chaque epoch
		#fichier.write(str("%0.3f"%batch_err)+",")
		#fichier.close()
		"""