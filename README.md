# Projet AIOI

https://www.kaggle.com/c/stanford-covid-vaccine

Ce projet à pour but de prédire les taux de dégradation probables pour chaque base d'ARN donnés. Des méthodes de deep learning ont été appliquées pour mettre en place différents types de modèles qui seront ensuite utilisés pour réaliser les prédictions.

## Résultats

Le dossier **./Models** contient les différents que nous avons obtenu:

- Keras_models: les poids (déterminer après le fit) et l'architecture des modèles sauvegarder dans des fichiers *.h5*
- Neural_network: les plots de l'architecture des différents modèles
- Optimisation: plot de la loss & val loss pour un ensemble de decay rate pour chaque modèle
- Submission: les targets prédits
- Summarize_models: plot de la loss & mse de chaque modèle, réalisé après optimisation & validation
- kfold_validation_scores.txt: Ecart type et moyenne de la loss et du mse obtenu après une étape de repeated k-fold cross-validation pour chaque modèle 
- models_history.npy: sauvegarde de l'history (loss & mse) de chaque modèle, obtenu après optimisation & validation

## Prérequis

L'utilisation de [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) est fortement recommandée pour l'utilisation du programme de Deep Learning.

## Quick start

1. Clone du répertoire github

> Lien HTTPS

```
git clone https://github.com/Dylkln/Projet_AIOI.git
```

> Lien SSH

```
git clone git@github.com:Dylkln/Projet_AIOI.git
```

2. Initialiser l'environnement conda à partir du fichier *environment.yml*

```
conda env create --file environment.yml
```

3. Activer l'environnement conda

```
conda activate deep-learning
```

## Auteurs

BODIN Raphaël: bodin.raphael85@gmail.com

IMBERT Pierre

KLEIN Dylan: klein.dylan@outlook.com

MAURY Léo: leomaury97@gmail.com

PICHON Julien: julien.pichon@cri-paris.org

Université de Paris M2-BI

## Date

23 octobre 2020
