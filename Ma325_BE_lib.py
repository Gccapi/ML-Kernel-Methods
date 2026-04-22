# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 14:56:36 2025

@author: floal
"""
#%% chapitre 1 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as r
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def creation_point(n , choix=0):
    X = np.zeros((1,n))
    Y = np.zeros((1,n)) 
    lim = int(n/2)
    if choix ==0:
        for i in  range(lim):
            X[0,i] = r.randint(-100, 100)
            Y[0,i] = r.randint(-100, -10)
        for i in  range(lim , n):
            X[0,i] = r.randint(-100, 100)
            Y[0,i] = r.randint(10, 100)
        return X ,Y
    if choix ==1:
        lambda_param_1 = 0.5
        lambda_param_2 = 0.01
        valeurs_1 = np.random.exponential(scale=1/lambda_param_1, size=lim)  # Générer des valeurs exponentielles         
        valeurs_clippees_1 = np.clip(valeurs_1, -100, 100)
        valeurs_2 = np.random.exponential(scale=1/lambda_param_1, size=lim)  # Générer des valeurs exponentielles
        valeurs_clippees_2 = np.clip(valeurs_2, -100, 100)
        X[0,:lim] = valeurs_clippees_1
        Y[0,:lim] = valeurs_clippees_2

 
        valeurs_1 = np.random.exponential(scale=1/lambda_param_2, size=lim)  # Générer des valeurs exponentielles
        valeurs_clippees_1 = np.clip(valeurs_1, -100, 100)
        valeurs_2 = np.random.exponential(scale=1/lambda_param_2, size=lim)  # Générer des valeurs exponentielles
        valeurs_clippees_2 = np.clip(valeurs_2, -100, 100)
        X[0,lim:] = valeurs_clippees_1
        Y[0,lim:] = valeurs_clippees_2
        return X ,Y
    if choix ==2:
        mu_1 = -20
        mu_2 = 20
        var_1 = 20
        var_2 = 20
        for i in  range(lim):
            valeurs_1 = r.gauss(mu_1, var_1)  # Générer des valeurs exponentielles         
            valeurs_clippees_1 = np.clip(valeurs_1, -100, 100)
            valeurs_2 = r.gauss(mu_1, var_1)   # Générer des valeurs exponentielles
            valeurs_clippees_2 = np.clip(valeurs_2, -100, 100)
            X[0,i] = valeurs_clippees_1
            Y[0,i] = valeurs_clippees_2
        for i in  range(lim , n):
            valeurs_1 = r.gauss(mu_2, var_2)  
            valeurs_clippees_1 = np.clip(valeurs_1, -100, 100)
            valeurs_2 = r.gauss(mu_2, var_2)  
            valeurs_clippees_2 = np.clip(valeurs_2, -100, 100)
            X[0,i] = valeurs_clippees_1
            Y[0,i] = valeurs_clippees_2
        return X ,Y
    if choix ==3:
        R_1 = 5*np.random.uniform(size=lim) + 5
        R_2 = 4*np.random.uniform(size=lim) + 16
        theta = np.linspace(0, 2*np.pi , lim)
        X[0 , :lim] = R_1 * np.cos(theta)
        Y[0 , :lim] = R_1 * np.sin(theta)
        X[0 , lim:] = R_2 * np.cos(theta)
        Y[0 , lim:] = R_2 * np.sin(theta)
        return X ,Y
 
    
 
def ACP(data):
    m,n = np.shape(data)
    a = np.mean(data,axis=0)
 
    Ac = data - a
 
    D = np.zeros((n,n))
    print("ca roule")
    for i in range(n):
        D[i,i] = np.sqrt( np.mean(Ac[:,i]**2))    
    D = D + 10**(-6)
    Acr = Ac@np.linalg.inv(D)
 
    U , S , VT = np.linalg.svd(Acr)
    P = np.zeros((m,n))
    taille = len(S)
    C=(1/(m*n))* VT.T @ np.diag(S**2)
    L=(1/(m*n))*S**2 
    for i in range(taille):
        P[:,i] = U[:,i]*S[i]
    return P , L ,C
 

#%% chapitre 2 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def noyau_gaussien(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))

def noyau_exponentiel(x, y):
    return np.exp(np.dot(x, y))

def noyau_polynomial(x, y, c=1, d=2):
    return (np.dot(x, y) + c)**d

def distance_noyau(k, x0, y):
    return k(x0, x0) + k(y, y) - 2 * k(x0, y)

def visualiser_distance(nom, noyau, X,Y,x0):
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            y = np.array([X[i, j], Y[i, j]])
            Z[i, j] = distance_noyau(noyau, x0, y)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.viridis)
    ax.set_title(f"Distance dans l'espace de redescription ({nom})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Distance")
    plt.tight_layout()
    plt.show()

#%% chapitre 3
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score

# Fonction pour construire la matrice de Gram à partir d'un noyau
def compute_kernel_matrix(X, kernel_func):
    m = X.shape[0]
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = kernel_func(X[i], X[j])
    return K

# Exemple de noyau gaussien (RBF)
def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))

# Algorithme des moindres carrés à noyaux multiclasse
class MCNMulticlass:
    def __init__(self, kernel_func, lambda_reg=1e-4):
        self.kernel_func = kernel_func
        self.lambda_reg = lambda_reg
        self.X_train = None
        self.A = None
    
    def fit(self, X, y):
        self.X_train = X
        m = X.shape[0]
        c = len(np.unique(y))  # Nombre de classes
        
        # Construction de la matrice de labels Y (m x c)
        Y = np.zeros((m, c))
        for i in range(m):
            Y[i, y[i]] = 1
        
        # Calcul de la matrice de Gram K
        K = compute_kernel_matrix(X, self.kernel_func)
        
        # Ajout de la régularisation et résolution pour A
        I = np.eye(m)
        self.A = np.linalg.solve(K + self.lambda_reg * I, Y)
    
    def predict(self, X_test):

        m_test = X_test.shape[0]
        m_train = self.X_train.shape[0]
        
        # Calcul du vecteur de noyau pour chaque échantillon de test
        K_test = np.zeros((m_test, m_train))
        for i in range(m_test):
            for j in range(m_train):
                K_test[i, j] = self.kernel_func(X_test[i], self.X_train[j])
        
        # Prédiction
        Y_pred = K_test @ self.A
        return np.argmax(Y_pred, axis=1)

def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))

def compute_kernel_matrix(X, kernel_func):

    m = X.shape[0]
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = kernel_func(X[i], X[j])
    return K

def kernel_ridge_regression(X, y, kernel_func, lambda_reg=1e-4):

    K = compute_kernel_matrix(X, kernel_func)
    m = X.shape[0]
    I = np.eye(m)
    a = np.linalg.solve(K + lambda_reg * I, y)
    return a, X

def predict_kernel_ridge(X_test, X_train, a, kernel_func):

    m_test = X_test.shape[0]
    m_train = X_train.shape[0]
    K_test = np.zeros((m_test, m_train))
    for i in range(m_test):
        for j in range(m_train):
            K_test[i, j] = kernel_func(X_test[i], X_train[j])
    y_pred = K_test @ a
    return y_pred


def kernel_lasso_regression(X, y, kernel_func, lambda_reg=1e-4):
    K = compute_kernel_matrix(X, kernel_func)
    m = X.shape[0]
    a = cp.Variable(m)
    objective = cp.Minimize(0.5 * cp.sum_squares(K @ a - y) + lambda_reg * cp.norm1(a))
    problem = cp.Problem(objective)
    problem.solve()
    return a.value, X

#%% Chapitre 4

from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from PIL import Image
import os

def generate_data():
    return make_circles(n_samples=400, noise=0.05, factor=0.3, random_state=42)

def compute_variance_and_separation(X, y, gammas):
    variances = []
    separations = []
    for gamma in gammas:
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
        X_proj = kpca.fit_transform(X)
        
        var = np.var(X_proj[:, 0])
        variances.append(var)
        
        class0 = X_proj[y == 0]
        class1 = X_proj[y == 1]
        sep = np.linalg.norm(class0.mean(axis=0) - class1.mean(axis=0))
        separations.append(sep)

    return np.array(variances), np.array(separations)

def plot_variance_and_separation(gammas, variances, separations, gamma_optimal):
    plt.figure(figsize=(10, 6))
    plt.semilogx(gammas, variances, 'b-', alpha=0.5, label='Variance expliquée')
    plt.semilogx(gammas, separations, 'g-', alpha=0.5, label='Séparation des classes')
    plt.semilogx(gammas, 0.7 * variances + 0.3 * separations, 'k-', linewidth=2, label='Score combiné')
    plt.axvline(gamma_optimal, color='r', linestyle='--', label=f'γ optimal = {gamma_optimal:.2f}')
    plt.xlabel('Valeur de γ (échelle log)')
    plt.ylabel('Valeurs brutes')
    plt.title('Optimisation du paramètre γ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def kernel_pca_projection(X, gamma):
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
    return kpca.fit_transform(X)

def plot_pca_comparisons(X, y, gamma_optimal, gamma_suboptimal):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f8f8f8')

    # Projection sous-optimal
    X_sub = kernel_pca_projection(X, gamma_suboptimal)
    ax1.scatter(X_sub[:, 0], X_sub[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax1.set_title(f"γ = {gamma_suboptimal} (sous-optimal)", pad=12)

    # Projection optimal
    X_opt = kernel_pca_projection(X, gamma_optimal)
    ax2.scatter(X_opt[:, 0], X_opt[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax2.set_title(f"γ optimal = {gamma_optimal:.2f}", pad=12)

    for ax in (ax1, ax2):
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Fonction pour créer des données spirales
def generate_spiral_data():
    theta = np.linspace(0, 4 * np.pi, 1000)
    r = np.linspace(0.5, 1.5, 1000)
    X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return X, theta

# Fonction de visualisation ACP classique et ACP noyau (spirale)
def plot_spiral_comparison(X, theta, X_pca, X_kpca, pca, gamma_optimal):
    plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

    # Données brutes
    ax0 = plt.subplot(gs[0])
    sc0 = ax0.scatter(X[:, 0], X[:, 1], c=theta, cmap='hsv', alpha=0.7)
    ax0.set_title("Données brutes\nSpirale 2D")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    plt.colorbar(sc0, ax=ax0, label='Angle θ')

    # ACP classique
    ax1 = plt.subplot(gs[1])
    sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=theta, cmap='hsv', alpha=0.7)
    ax1.set_title("ACP Linéaire\nVariance expl.: {:.1%}".format(pca.explained_variance_ratio_[0]))
    ax1.set_xlabel("Composante 1")
    ax1.set_ylabel("Composante 2")
    plt.colorbar(sc1, ax=ax1, label='Angle θ')

    # ACP noyau
    ax2 = plt.subplot(gs[2])
    sc2 = ax2.scatter(X_kpca[:, 0], X_kpca[:, 1], c=theta, cmap='hsv', alpha=0.7)
    ax2.set_title(f"ACP à Noyau (γ={gamma_optimal})\nStructure linéarisée")
    ax2.set_xlabel("Composante 1")
    ax2.set_ylabel("Composante 2")
    plt.colorbar(sc2, ax=ax2, label='Angle θ')

    plt.tight_layout()
    plt.show()

# Fonction pour optimiser gamma
def optimize_gamma(X, gammas):
    variances = []
    for g in gammas:
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=g)
        X_temp = kpca.fit_transform(X)
        var = np.var(X_temp[:, 0]) / np.sum(np.var(X_temp, axis=0))
        variances.append(var)
    return variances

# Fonction pour créer une animation
def create_animation(gammas, X, theta, output_filename):
    fig, ax = plt.subplots(figsize=(7, 5))

    def update(gamma):
        ax.clear()
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
        X_kpca = kpca.fit_transform(X)
        sc = ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=theta, cmap='hsv', alpha=0.7)
        ax.set_title(f"ACP à Noyau (γ = {gamma:.2f})")
        return sc,

    ani = FuncAnimation(fig, update, frames=gammas, blit=True)
    ani.save(output_filename, writer='pillow', fps=3)
    plt.close()

# Fonction pour convertir une animation GIF
def convert_gif_to_frames(input_filename, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    gif = Image.open(input_filename)
    for i in range(gif.n_frames):
        gif.seek(i)
        gif.save(f"{output_folder}/frame_{i:02d}.png")

#%% chapitre 5 

def SVM_uzawa(u, v, gamma, gamma_b, max_iter, tol ,  p , q):
    # Initialisation
    n = 2   # dimension
    lambda_vals = np.zeros(p)  # multiplicateurs pour les contraintes positives
    mu_vals = np.zeros(q)  # multiplicateurs pour les contraintes négatives
    b_tilde = 0.0  # biais initial
    w_tilde = np.zeros(n)  # initialisation de w_tilde
 
    L = [2 * tol, np.sum(lambda_vals) + np.sum(mu_vals)]
    k = 0
    # Algorithme d’Uzawa
    while np.linalg.norm(L[0] - L[1]) > tol and k < max_iter:
        k += 1
        # Mise à jour des variables primales à partir des multiplicateurs actuels
        w_tilde = np.sum(lambda_vals.reshape(-1, 1) * u, axis=0) - np.sum(mu_vals.reshape(-1, 1) * v, axis=0)
        # Mise à jour de b_tilde par descente sur la contrainte stationnaire : sum(lambda) = sum(mu)
        b_tilde = b_tilde - gamma_b * (np.sum(lambda_vals) - np.sum(mu_vals))
 
        # Mise à jour des multiplicateurs par gradient ascendant (avec projection sur R+)
        for i in range(p):
            delta_lambda = 1 - u[[i], :] @ w_tilde + b_tilde
            lambda_vals[i] = np.maximum(np.zeros(1), lambda_vals[i] + gamma * delta_lambda)[0]
        for j in range(q):
            delta_mu = 1 + v[[j], :] @ w_tilde - b_tilde
            mu_vals[j] = np.maximum(np.zeros(1), mu_vals[j] + gamma * delta_mu)[0]
        L[0], L[1] = L[1], np.sum(lambda_vals) + np.sum(mu_vals)
        print(np.linalg.norm(L[0] - L[1]))
    print("w_tilde =", w_tilde)
    print("b_tilde =", b_tilde)
    print("lambda =", lambda_vals)
    print("mu =", mu_vals)
    if n == 2:
        # Génération de la ligne x pour les hyperplans
        x_line = np.linspace(-100, 100, 200)
        # Hyperplan décisionnel
        y_decision = (b_tilde - w_tilde[0] * x_line) / w_tilde[1]
        # Hyperplan de marge positive
        y_margin_pos = (b_tilde + 1 - w_tilde[0] * x_line) / w_tilde[1]
        # Hyperplan de marge négative
        y_margin_neg = (b_tilde - 1 - w_tilde[0] * x_line) / w_tilde[1]
        # Création de la figure
        plt.figure(figsize=(8, 6))
        # Affichage des points positifs et négatifs
        plt.scatter(u[:, 0], u[:, 1], color='blue', label='Positifs')
        plt.scatter(v[:, 0], v[:, 1], color='red', label='Négatifs')
        # Tracé des hyperplans
        plt.plot(x_line, y_decision, 'k-', label="Hyperplan décisionnel")
        plt.plot(x_line, y_margin_pos, 'g--', label="Marge (+1)")
        plt.plot(x_line, y_margin_neg, 'g--', label="Marge (-1)")
        # Ajustements et légendes
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("SVM linéaire résolu par l’algorithme d’Uzawa avec marge rigide")
        plt.legend()
        plt.show()
    return w_tilde, b_tilde, lambda_vals, mu_vals
 
def SVM_noyau_gaussien( X , y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
    # Création du modèle SVM avec noyau gaussien (RBF)
    clf = svm.SVC(kernel='rbf', C=1.0, gamma=0.001)
 
    # Entraînement du modèle
    clf.fit(X_train, y_train)
 
    # Évaluation du modèle
    accuracy = clf.score(X_test, y_test)
    print(f"Précision du modèle : {accuracy:.2f}")
 
    # Visualisation des données et de la frontière de décision
    plt.figure(figsize=(8, 6))
 
    # Création d'une grille pour visualiser la frontière
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
 
    # Prédictions sur la grille
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
 
    # Affichage de la frontière de décision
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
 
    # Affichage des points de données
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title("SVM avec noyau gaussien (RBF)")
    plt.xlabel("Caractéristique 1")
    plt.ylabel("Caractéristique 2")
    plt.show()