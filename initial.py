#import matplotlib.pyplot as plt
import numpy as np
#historique de l'erreur par chaque thread
historique_split = []

#vecteur des gradients de chaque processus
grads = []
#nombre de travailleurs
nbre_processus = 20
#45% de travailleurs byzantins
nbre_byzantin = 6
#historique de l'erreur par chaque thread
#gradient utilisé par les travailleurs pour mettre à jour leur paramètre
grad = None
epochs = 500
#partitionner le dataset en t partition
t = 0
valeur_batch = 3

	
def mini_batch(liste1,liste2, batch):
	if batch == 0:
		return liste1, liste2
	else :
		x = []
		mini_b = []
		x2 = []
		mini_b2 = []
		for i in range(len(liste1)):
			x.append(liste1[i])
			x2.append(liste2[i])
			if (len(x) == batch):
				mini_b.append(x)
				x = []
				mini_b2.append(x2)
				x2 = []
		return np.array(mini_b),np.array(mini_b2)

#les byzantins
def byzantin(nbre_byzantin):
	byzantins = []
	for b in range(nbre_byzantin):
		#distribution gaussinene
		#mu1 ,sigma1 = -1,-0.6
		mu2, sigma2 = 0.6,1
		gradient1 = np.random.uniform(mu2,sigma2,(25,10))
		gradient2 = np.random.uniform(mu2,sigma2,(25,10))
		gradient_byzantin = np.vstack((gradient1,gradient2))
		gradient_byzantin = gradient_byzantin.ravel()
		np.random.shuffle(gradient_byzantin)
		gradient_byzantin = gradient_byzantin.reshape(50,10)
		byzantins.append(gradient_byzantin)
	return byzantins