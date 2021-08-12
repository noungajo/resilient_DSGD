import numpy as np

class tableaux():
	def __init__(self,taille):
		self.tab = np.random.randint(1,20,taille)
	def modif_tab(self):
		self.tab = 2*self.tab
	def affiche(self):
		print("mon tableau", self.tab)
		
tab1 = tableaux(10)
tab1.affiche()
tab1.modif_tab()
tab1.affiche()

mu1 ,sigma1 = -1,-0.6
mu2, sigma2 = 0.6,1
gradient1 = np.random.uniform(mu1,sigma1,(10,5))
gradient2 = np.random.uniform(mu2,sigma2,(10,5))
gradient_byzantin = np.vstack((gradient1,gradient2))
print(gradient_byzantin)
gradient_byzantin = gradient_byzantin.ravel()
np.random.shuffle(gradient_byzantin)
gradient_byzantin = gradient_byzantin.reshape(20,5)
print(list(gradient_byzantin))