#apprentissage distribué
ceci est le code source python d'un algorithme centralisé qui mime l'apprentissage distribué

## que fait ce code
- charger l'ensemble de données (MNIST dataset)
- créer n tableaux en associant à chaque tableau une partie distincte de l'ensemble de données
- créer n tableaux en associant à chaque tableau un réseau de neuronnes et chaque élément du tableau va estimer 
son gradient et l'insérer dans une liste appelé grads
- une fois que la liste de gradient atteint n, on agrège et lance la phase de mise à jour de chaque élément du tableau

## les fichiers qu'il contient
- activation : contient la fonction d'activation(tanh)
- agregation: contient les règles d'agrégation (krum, average, median, trimmed mean, FABA, Bulyan, Variante 1, Variante 2, Variante 3)
- couche : contient les caractéristiques communes à toute les autres couches(entrée,sortie), une fonction qui 
fait la propagation avant et une autre fonction qui effectue la rétropropagation du gradient
- couche_activation : contient des fonctions non linéaires à appliquer à la sortie de certaines couches.
- FC : premier type de couche, Les couches FC sont les couches les plus élémentaires car tous les
neurones d’entrée sont connectés à tous les neurones de sortie
- mnist : fichier d'exécution pour charger le dataset, créer les n réseaux de neurones et lancer la phase d'apprentissage
- network : permet de construire le réseau de neuronnes
- perte : contient la fonction perte 
- initial : contient les valeurs initiales à savoir le nombre de workers, le nombre de noeuds byzantins, le nombre d'époque et la taille du batch

## initialisation le nombre de workers
- pour modifier le nombre de workers il faut le faire dans le fichier initial.py en modifiant la variable nbre_processus
- pour spécifier le nombre de workers byzantins il faut modifier la variable "nbre_byzantin" contenue dans le fichier initial.py

## choisir la règle d'agrégation
pour choisir la règle d'agrégation il faut aller dans le fichier mnist.py changer la règle d'agrégation par celle qui a été importée

## Comment exécuter ce code
- installer python 3.x
- installer keras, numpy
- ouvrir un terminal
- se positionner dans le repertoire courant taper la commande: python minst.py 
- pour la première utilisation une connexion internet est necessaire car le dataset est chargé en ligne depuis un serveur distant
- durant l'exécution le terminal va montrer les différentes valeur de l'erreur en fonction de l'époque pour un worker

## le résultat
- au niveau du terminal on peut observer la variation de l'erreur
- dans le répertoire historique_erreur on y trouve des fichiers
- chaque fichier contient les valeurs de l'erreur de 3 workers à chaque époque
- une fois ces données fourni par le programme, on les extraits pour construire les courbes.
NB: lors des expérimentation le batch était manipulé manuellement, les résultats obtenus copiés pour la représentation graphique