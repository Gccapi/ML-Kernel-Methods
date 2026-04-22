# Advanced Machine Learning : Theory and Practice of Kernel Methods

Ce projet explore l'**Astuce des Noyaux (Kernel Trick)**, une technique fondamentale permettant d'étendre les algorithmes d'apprentissage linéaire à la résolution de problèmes non-linéaires complexes.

## Objectifs du Projet
L'enjeu est de projeter des données non-linéairement séparables dans un espace de Hilbert de dimension supérieure où une séparation linéaire devient possible, sans pour autant subir l'explosion du coût computationnel.


## Concepts & Algorithmes Implémentés
- **Algorithme d'Uzawa** : Implémentation "from scratch" d'un solveur pour le problème d'optimisation sous contraintes (marge rigide des SVM).
- **Noyaux de type Positif** : Étude et application des noyaux Gaussiens (RBF), polynomiaux et sigmoïdes.
- **ACP à Noyaux (Kernel PCA)** : Extension de l'Analyse en Composantes Principales pour la réduction de dimension non-linéaire.
- **Moindres Carrés à Noyaux** : Généralisation des méthodes de régression et d'interpolation.

## Études de Cas & Datasets
Le projet valide les modèles sur des jeux de données variés, allant de simulations statistiques à des bases de données réelles :
- **Lois statistiques** : Distributions normales et exponentielles.
- **Iris Dataset** : Classification classique de fleurs.
- **Fashion-MNIST** : Reconnaissance de vêtements (Computer Vision) pour tester la robustesse des noyaux sur de grandes dimensions.

## Documentation
Le dépôt inclut un rapport technique complet de 38 pages (`BE_MA325.pdf`) détaillant les démonstrations mathématiques (Théorème de Mercer, inégalité de Cauchy-Schwarz) et les analyses de performances.

## Auteurs
- Florian Alaux
- Romain Lefol
- Simon Lignac
- Steeve Mbock
- Guilhem Perret-Bardou
- Nahia Zabala
