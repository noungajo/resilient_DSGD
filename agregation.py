import numpy as np
from initial import nbre_byzantin, nbre_processus

def average(grads):
    return sum(grads)/len(grads)

#V1 et V2 doivent etre des arrays et non des liste
def distance(v1, v2): 
    return np.sqrt(np.sum((v1 - v2) ** 2)) 
	
def krum(grads):
    f = nbre_byzantin
    if nbre_processus != len(grads):
        print("difference entre grads et nbre")
    gradient = np.zeros(len(grads))
    for i in range(len(grads)):
        distances = np.zeros(len(grads))
        #distance entre i et les autres vecteurs
        for j in range(len(grads)):
            if i != j:
                distances[j]=distance(grads[i],grads[j])
        distance_mins = np.zeros(len(grads) - f - 2)
        #determiner les n-f-2 plus proche vecteurs de i
        for j in range(len(grads) - f - 2):
            distance_mins[j] = distances.min()
            #indice = distance.argmin()
            distances[distances.argmin()] = distances.max()
        
        gradient[i] = distance_mins.sum()
    return gradient.argmin()

def median(grads):
    index = 0
    M = np.zeros((grads[index].shape[0],grads[index].shape[1]))
    for ligne in range(grads[index].shape[0]):
        for colonne in range(grads[index].shape[1]):
            element_i_de_chaque_worker = []
            for j in range(len(grads)):
                element_i_de_chaque_worker.append(grads[j][ligne][colonne])
            M[ligne][colonne] = np.median(np.array(element_i_de_chaque_worker))
    return M
	
#version originale
def faba(grads):
	k = 1
	if nbre_processus != len(grads):
		print("difference entre grads et nbre")
	
	#distance entre la moyenne les differents gradient
	
	if nbre_byzantin==0:
		return average(grads)
	else :
		while k < nbre_byzantin:
			g0 = average(grads)
			difference = []
			for i in range(len(grads)):
				difference.append(distance(g0,grads[i]))
		#elimination du gradient le plus éloigné de g0
		
			index_max = np.array(difference).argmax()
			new_grads = []
			for j in range(len(difference)):
				if j != index_max :
					new_grads.append(grads[j])
			grads = new_grads
			k = k +1
		#retourner la moyenne de ces vecteurs
		return average(grads)
		
		
def bulyan(grads):
    select = []
    save = grads
    theta = len(grads)-2*nbre_byzantin
    if not (nbre_processus >= 4*nbre_byzantin + 3):
        return median(grads)
    else:  
        for j in range(theta):
            index = krum(save)
            temps = []
            for i in range(len(save)):
                if i != index:
                    temps.append(save[i])
            select.append(save[index])
            save = temps
        return median(select)
		
		
	

		
	
def trimmed(grads):
    index = 0
    betha = nbre_byzantin
    M = np.zeros((grads[index].shape[0],grads[index].shape[1]))
    for ligne in range(grads[index].shape[0]):
        for colonne in range(grads[index].shape[1]):
            element_i_de_chaque_worker = []
            for j in range(len(grads)):
                element_i_de_chaque_worker.append(grads[j][ligne][colonne])
            m_trie = sorted(element_i_de_chaque_worker)
            m_selectionne = m_trie[betha:-betha]
            d = len(grads) - 2*betha
            gk = (1/d)*sum(m_selectionne)
            M[ligne][colonne]=gk
    return M
		
def variante_1(grads):
	#seuil = int(0.2*nbre_processus)
	plage = len(grads) - nbre_byzantin - 2
	if nbre_processus != len(grads):
		print("difference entre grads et nbre")
	g0 = average(grads)
	#distance entre la moyenne les differents gradient
	distances = np.zeros(len(grads))
	if nbre_byzantin==0:
		return g0
	else :
		for i in range(len(grads)):
			distances[i] = distance(grads[i], g0)
		#elimination des f gradients plus loin de la moyenne
		distance_mins = np.zeros(plage + 1,dtype=int)
		for j in range(plage + 1):
			distance_mins[j] = distances.argmin()
			distances[distances.argmin()] = distances.max()
		resultat = np.array(grads)[distance_mins]
		#retourner la moyenne de ces vecteurs
		return average(resultat)
		
def variante_2(grads):
    index = 0
    nbre_supprime = 0
    #matrice dont les éléments sont la la médiane de la coordonnée j de chaque worker
    M = np.zeros((grads[index].shape[0],grads[index].shape[1]))
    for ligne in range(grads[index].shape[0]):
        for colonne in range(grads[index].shape[1]):
            #extraction de la coordonnée i de chaque worker
            element_i_de_chaque_worker = []
            for j in range(len(grads)):
                element_i_de_chaque_worker.append(grads[j][ligne][colonne])
                
            #liste de gradient temporaire
            index_del = []
            q1,q3 = np.percentile(element_i_de_chaque_worker,[25,75])
            cond1 = q3 +1.5*(q3-q1)
            cond2 = q1 - 1.5*(q3-q1)
            for j in range(len(grads)):
                if grads[j][ligne][colonne] > cond1 or grads[j][ligne][colonne] < cond2:
                    #recensement des index à delete
                    index_del.append(j)
                    nbre_supprime = nbre_supprime + 1
            #suppression des index recensés
            for element in index_del:
                del grads[index]
    return average(grads), nbre_supprime
	
	
	
def variante_3(grads):
    index = 0
    nbre_supprime = 0
	
    #matrice dont les éléments sont la la médiane de la coordonnée j de chaque worker
    u,s = np.zeros((grads[index].shape[0],grads[index].shape[1])),np.zeros((grads[index].shape[0],grads[index].shape[1]))
    for ligne in range(grads[index].shape[0]):
        for colonne in range(grads[index].shape[1]):
            #extraction de la coordonnée i de chaque worker
            element_i_de_chaque_worker = []
            for j in range(len(grads)):
                element_i_de_chaque_worker.append(grads[j][ligne][colonne])
                
            u[ligne][colonne],s[ligne][colonne] = np.array(element_i_de_chaque_worker).mean(),np.array(element_i_de_chaque_worker).std()
     #copie de la liste des gradients
    grad2 = grads.copy()       
    colonne = 0
    ligne = 0
    for j in range(len(grads)):
        while ligne < grads[index].shape[0]:
            while colonne < grads[index].shape[1]:
                g = (grads[j][ligne][colonne] - u[ligne][colonne])/s[ligne][colonne]
                dif = len(grads) - len(grad2)
                if abs(g) > 3:
                #supression
                    if dif == 0:
                        del grad2[j]
                    else:
                        del grad2[j-dif]
                    nbre_supprime = nbre_supprime + 1
                    colonne = grads[index].shape[1]
                    ligne = grads[index].shape[0]	
                colonne = colonne + 1
            ligne = ligne + 1				
    return average(grad2), nbre_supprime

