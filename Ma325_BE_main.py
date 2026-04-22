# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 14:56:07 2025

@author: floal
"""

import matplotlib.pyplot as plt
import cvxpy as cp
from Ma325_BE_lib import *
import numpy as np
import pandas as pd
import random as r
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#%% Chapitre 1
#ACP sur des nuage de point suivant une loi uniforme
n=1000
lim = int(n/2)
x , y = creation_point(n,0)
data = np.vstack((x,y))
data = data.T@data
 
plt.figure()
plt.plot( x[0,0:lim],y[0,0:lim] , ".b")
plt.plot( x[0,lim:],y[0,lim:] , ".r")
plt.title("jeux de données")
 
 
P0 , L , C = ACP(data)
t = np.arange(0,lim, 1 )
plt.figure()
plt.plot(t,P0[:lim,0] , ".b" ,label="type 1")
plt.plot(t,P0[lim:,0], ".r", label="type 2")
 
 
plt.ylabel("taux explicatif {}%".format(100*L[0]))
plt.title("ACP sur des nuages de points suivant une loi uniforme")
#ACP sur une loi normal
n=1000
lim = int(n/2)
x , y = creation_point(n,2)
data = np.vstack((x,y))
data = data.T@data
 
plt.figure()
plt.plot( x[0,0:lim],y[0,0:lim] , ".b")
plt.plot( x[0,lim:],y[0,lim:] , ".r")
plt.title("jeux de données")
 
P0 , L , C = ACP(data)
t = np.arange(0,lim, 1 )
plt.figure()
plt.plot(t,P0[:lim,0] , ".b" ,label="type 1")
plt.plot(t,P0[lim:,0], ".r", label="type 2")
 
 
plt.ylabel("taux explicatif {}%".format(100*L[0]))
plt.title("ACP sur une loi normal")
plt.legend()
 
 
#ACP sur iris
data  = pd.read_csv("Data/iris.csv")
data = data.to_numpy()
data = data[:,:4]
 
P0 , L , C = ACP(data)
 
plt.figure()
plt.plot(P0[:50,0],P0[:50,1] , ".b" ,label="espece 0")
plt.plot(P0[51:100,0],P0[51:100,1] , ".r", label="espece 1")
plt.plot(P0[101:150,0],P0[101:150,1]  , ".y" , label="espece 2")
 
plt.xlabel("taux explicatif {}%".format(100*L[0]))
plt.ylabel("taux explicatif {}%".format(100*L[1]))
plt.title("ACP sur iris")
plt.legend()
 
#ACP sur fashion-mnist_train
data  = pd.read_csv("Data/fashion-mnist_train.csv")
data = data.to_numpy()
data = data[data[:,0].argsort()]
repetition = {}
for i in [0,1,2,3,4,5,6,7,8,9]:
    S=0
    for j in data[:,0]:
        if i==j:
            S=S+1
    repetition[i]=S
data = data[:,1:]
data_0 = data[:6000,:]
data_1 = data[6001:12000,:]
data_2 = data[12001:18000,:]
 
data_test = np.vstack((data_0[:600,:],data_1[:600,:],data_2[:600,:]))
 
P0 , L , C = ACP(data_test)
 
plt.figure(figsize=(10, 10))
plt.plot(P0[:600,0],P0[:600,1] , ".b" ,label="les 0")
plt.plot(P0[601:1200,0],P0[601:1200,1] , ".r", label="les 1")
plt.plot(P0[1201:1800,0],P0[1201:1800,1]  , ".y" , label="les 2")
 
plt.xlabel("taux explicatif {}%".format(100*L[0]))
plt.ylabel("taux explicatif {}%".format(100*L[1]))
plt.title("ACP sur fashion mnist")
plt.legend()
 
fig = plt.figure(figsize=(10, 10))  # Taille de la figure
ax = fig.add_subplot(111, projection='3d')
 
 
ax.scatter(P0[:600, 0], P0[:600, 1], P0[:600, 2],  # Coordonnées x, y, z
           color='blue', alpha=0.7, s=20)  # Style des points : couleur, transparence, taille
 
ax.scatter(P0[601:1200, 0], P0[601:1200, 1], P0[601:1200, 2],  
           color='red')
ax.scatter(P0[1201:1800, 0], P0[1201:1800, 1], P0[1201:1800, 2],  
           color='yellow', alpha=0.7, s=20)  
 
# Configurer l'affichage des axes
 
ax.set_xlabel("taux explicatif {}%".format(100*L[0]))
ax.set_ylabel("taux explicatif {}%".format(100*L[1]))
ax.set_zlabel("taux explicatif {}%".format(100*L[2]))
ax.set_title("ACP projection sur 3 axes")
 
#%% graphes distances pour chaque noyau dans l’espace de redescription pour [-2, 2]²
x0 = np.array([0.0, 0.0]) 
grid = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(grid, grid)

visualiser_distance("Noyau Gaussien", lambda x, y: noyau_gaussien(x, y, sigma=0.5),X,Y,x0)
visualiser_distance("Noyau Exponentiel", noyau_exponentiel,X,Y,x0)
visualiser_distance("Noyau Polynomial", lambda x, y: noyau_polynomial(x, y, c=1, d=2),X,Y,x0)

#%% Test avec données MNIST
# Charger les données MNIST depuis les fichiers CSV sans en-tête
train_data = pd.read_csv('Data/mnist_train.csv', header=None)
test_data = pd.read_csv('Data/mnist_test.csv', header=None)
    
    # Extraction des labels (première colonne) et des features (colonnes restantes)
y_train = train_data.iloc[:, 0].values  # Colonne 0 pour les labels
X_train = train_data.iloc[:, 1:].values  # Colonnes 1 à 784 pour les pixels
y_test = test_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
    
    # Normalisation des pixels (valeurs de 0 à 255 -> 0 à 1)
X_train = X_train / 255.0
X_test = X_test / 255.0
    
    # Limiter le nombre d'échantillons pour accélérer les calculs
max_train_samples = 1000
max_test_samples = 500
X_train = X_train[:max_train_samples]
y_train = y_train[:max_train_samples]
X_test = X_test[:max_test_samples]
y_test = y_test[:max_test_samples]
    
    # Initialisation du modèle avec noyau gaussien
sigma = 10.0  # Paramètre du noyau RBF
lambda_reg = 0.01  # Paramètre de régularisation
model = MCNMulticlass(kernel_func=lambda x, y: gaussian_kernel(x, y, sigma=sigma), 
                                        lambda_reg=lambda_reg)
    
    # Entraînement du modèle
print("Entraînement du modèle...")
model.fit(X_train, y_train)
 
    # Prédiction sur les données de test
print("Prédiction sur les données de test...")
y_pred = model.predict(X_test)
    
    # Calcul de la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision sur l'ensemble de test : {accuracy * 100:.2f}%")

#%% Chapitre 3d : Moindres carrés avec régularisation L2 à noyaux

# Génération de données synthétiques non linéaires
np.random.seed(42)
n_samples = 100
X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
y = np.sin(2 * X).ravel() + 0.1 * np.random.randn(n_samples)  # Signal bruité

# Paramètres
sigma = 0.5  # Paramètre du noyau gaussien
lambda_reg = 0.01  # Paramètre de régularisation

# Entraînement du modèle
a, X_train = kernel_ridge_regression(X, y, kernel_func=lambda x, y: gaussian_kernel(x, y, sigma=sigma), 
                                     lambda_reg=lambda_reg)

# Prédiction sur une grille pour visualisation
X_test = np.linspace(-2.5, 2.5, 200).reshape(-1, 1)
y_pred = predict_kernel_ridge(X_test, X_train, a, lambda x, y: gaussian_kernel(x, y, sigma=sigma))

# Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Données bruitées')
plt.plot(X_test, y_pred, color='red', label='Régression Ridge à noyaux')
plt.plot(X_test, np.sin(2 * X_test), color='green', linestyle='--', label='Vraie fonction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Régression Ridge à noyaux (noyau gaussien)')
plt.legend()
plt.grid(True)
plt.show()

# Analyse des résultats
print("Erreur quadratique moyenne sur les données d'entraînement :",
      np.mean((predict_kernel_ridge(X, X_train, a, lambda x, y: gaussian_kernel(x, y, sigma=sigma)) - y)**2))

#%% Chapitre 3d : Moindres carrés avec régularisation L1 à noyaux

# Entraînement du modèle Lasso
a_lasso, X_train = kernel_lasso_regression(X, y, lambda x, y: gaussian_kernel(x, y, sigma=sigma), 
                                           lambda_reg=0.1)

# Prédiction
y_pred_lasso = predict_kernel_ridge(X_test, X_train, a_lasso, lambda x, y: gaussian_kernel(x, y, sigma=sigma))

# Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Données bruitées')
plt.plot(X_test, y_pred_lasso, color='purple', label='Régression Lasso à noyaux')
plt.plot(X_test, np.sin(2 * X_test), color='green', linestyle='--', label='Vraie fonction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Régression Lasso à noyaux (noyau gaussien)')
plt.legend()
plt.grid(True)
plt.show()

# Analyse
print("Erreur quadratique moyenne (Lasso) :",
      np.mean((predict_kernel_ridge(X, X_train, a_lasso, lambda x, y: gaussian_kernel(x, y, sigma=sigma)) - y)**2))

#%% chapitre 4 valeur gamma

# Génération des données
X, y = generate_data()

gammas = np.logspace(-2, 2, 50)

# Calcul de la variance et séparation pour différents gamma
variances, separations = compute_variance_and_separation(X, y, gammas)

combined = 0.7 * variances + 0.3 * separations# Combinaison pondérée avec 70% variance + 30% séparation
gamma_optimal = gammas[np.argmax(combined)]

plot_variance_and_separation(gammas, variances, separations, gamma_optimal)


#%% comparaison gamma sous opti et opti 

gamma_suboptimal = 0.1
plot_pca_comparisons(X, y, gamma_optimal, gamma_suboptimal)


X_opt = kernel_pca_projection(X, gamma_optimal)
plt.figure(figsize=(8, 6))
plt.scatter(X_opt[:, 0], X_opt[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title(f'Projection avec γ optimal = {gamma_optimal:.2f}')
plt.xlabel('Composante 1'); plt.ylabel('Composante 2')
plt.colorbar(label='Classe')
plt.show()

#%% Comparaison ACP classique et noyau sur données spirale

X, theta = generate_spiral_data()

# Calcul des projections (ACP classique et ACP noyau)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
gamma_optimal = 15  # par validation croisée
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma_optimal)
X_kpca = kpca.fit_transform(X)

# Visualisation des trois résultats (brutes, ACP classique, ACP noyau)
plot_spiral_comparison(X, theta, X_pca, X_kpca, pca, gamma_optimal)


#%% Importance gamma 
gammas = np.logspace(-2, 2, 20)
variances = optimize_gamma(X, gammas)

plt.figure(figsize=(8, 4))
plt.semilogx(gammas, variances, 'o-')
plt.axvline(gamma_optimal, color='r', linestyle='--')
plt.title("Optimisation du paramètre γ")
plt.xlabel("Valeur de γ (échelle log)")
plt.ylabel("Variance expliquée (1ère composante)")
plt.grid(True)
plt.show()

# Animation en fonction de gamma
create_animation(gammas, X, theta, 'kpca_animation.gif')

# Conversion du gif pour latex en 30 frames
convert_gif_to_frames("kpca_animation.gif", "gamma_frames")

#%% Comparaison ACP classique et noyau sur cercles concentriques 

# on refait tout comme pour la spirale cette fois ci avec des données en cecles concentriques 
X, y = make_circles(n_samples=400, noise=0.05, factor=0.3, random_state=42)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=3)
X_kpca = kpca.fit_transform(X)

plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

# Données brutes
ax0 = plt.subplot(gs[0])
sc0 = ax0.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
ax0.set_title("Données brutes\nCercles concentriques")
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.grid(True, alpha=0.3)

# ACP classique
ax1 = plt.subplot(gs[1])
sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
ax1.set_title("ACP Linéaire\nVariance expl.: {:.1%}".format(pca.explained_variance_ratio_[0]))
ax1.set_xlabel("Composante 1")
ax1.set_ylabel("Composante 2")
ax1.grid(True, alpha=0.3)

# ACP noyau
ax2 = plt.subplot(gs[2])
sc2 = ax2.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', alpha=0.7)
ax2.set_title(f"ACP à Noyau (γ=3)\nDépliage optimal")
ax2.set_xlabel("Composante 1")
ax2.set_ylabel("Composante 2")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Animation pour cercles concentriques
gammas = np.logspace(-1, 1.8, 30)
create_animation(gammas, X, y, 'kpca_circles_animation.gif')

# Conversion pour latex
convert_gif_to_frames("kpca_circles_animation.gif", "gamma_frames1")

 #%% chapitre 5 
 #SVM sans noyau
  
 # Génération de données synthétiques pour l’exemple
np.random.seed(42)
p = 50  # nombre d’exemples positifs
q = 50  # nombre d’exemples négatifs
n = 2   # dimension
  
X , Y = creation_point(q+p, 0)
  
u = np.vstack((X[0,:q] , Y[0,:q]  )).T
v = np.vstack((X[0,p:] , Y[0,p:]  )).T
  
  
 # Paramètres de l’algorithme d’Uzawa
gamma = 0.0001  # pas pour la mise à jour des multiplicateurs
gamma_b = 0.0001  # pas pour la mise à jour de b
max_iter = 1000
tol = 1e-6
  
w_tilde, b_tilde, lambda_vals, mu_vals = SVM_uzawa(u, v, gamma, gamma_b, max_iter, tol , p , q )
  
 #SVM a noyau
  
 #application sur un nuage de point
nombre_de_point=100
lim = int(nombre_de_point/2)
X , Y = creation_point(nombre_de_point, 0)
  
u = np.vstack((X[0,:lim] , Y[0,:lim]  )).T
v = np.vstack((X[0,lim:] , Y[0,lim:]  )).T
  
U = np.vstack((u,v))
y = np.vstack( ( np.ones((lim,1)) , np.zeros((lim,1)) ) )
  
 # Mélanger les données avant de les diviser
U_shuffled, y_shuffled = U.copy(), y.copy()
indices = np.arange(len(y))  # Indices des échantillons
np.random.shuffle(indices)  # Mélange des indices
 
U_shuffled = U[indices]
y_shuffled = y[indices]
y_shuffled = np.squeeze(y_shuffled)
 
SVM_noyau_gaussien(U_shuffled , y_shuffled)
 
# application sur deux cercle
nombre_de_point=100
lim = int(nombre_de_point/2)
X , Y = creation_point(nombre_de_point, 3)
 
u = np.vstack((X[0,:lim] , Y[0,:lim]  )).T
v = np.vstack((X[0,lim:] , Y[0,lim:]  )).T
 
U = np.vstack((u,v))
y = np.vstack( ( np.ones((lim,1)) , np.zeros((lim,1)) ) )
 
# Mélanger les données avant de les diviser
U_shuffled, y_shuffled = U.copy(), y.copy()
indices = np.arange(len(y))  # Indices des échantillons
np.random.shuffle(indices)  # Mélange des indices
 
U_shuffled = U[indices]
y_shuffled = y[indices]
y_shuffled = np.squeeze(y_shuffled)
 
SVM_noyau_gaussien(U_shuffled , y_shuffled)