# Projet AIOI

https://www.kaggle.com/c/stanford-covid-vaccine

Ce projet à pour but de prédire les taux de dégradation probables pour chaque base d'ARN donnés. Des méthodes de deep learning ont été appliquées pour mettre en place différents types de modèles qui seront ensuite utilisés pour réaliser les prédictions.

## Résultats

Le jupyter notebook **corona_arn.ipynb** renseigne des **analyses** qui ont été menées à bien en détail et présente les **résultats**.

Le **module aioi** a été mis en place et permet de réaliser les réseaux de neurones, l'optimisation des modèles, de vérifier la validité des modèles (repeated k-fold cross-validation), l'apprentissage des modèles à partir d'arn_train, d'évaluer les performances des modèles et la prédiction à partir d'arn_test.

Le dossier **./Models** contient les différents que nous avons obtenu:

- Keras_models: les poids (déterminés après le fit) et l'architecture des modèles sauvegardés dans des fichiers *.h5*
- Neural_network: les plots de l'architecture des différents modèles
- Optimisation: plot de la loss & val loss pour un ensemble de decay rate pour chaque modèle
- Submission: les targets prédits
- Summarize_models: plot de la loss & mse de chaque modèle, réalisé après optimisation & validation
- kfold_validation_scores.txt: Ecart type et moyenne de la loss et du mse obtenu après une étape de repeated k-fold cross-validation pour chaque modèle 
- models_history.npy: sauvegarde de l'history (loss & mse) de chaque modèle, obtenu après optimisation & validation

On a également redéterminé les prédictions des structures & loop types (spotrna_train_good.tsv), l'apprentissage du modèle et son évaluation ont donc également été menés sur ces données et sont sauvegardés dans:

- New_keras_models: les poids (déterminés après le fit) et l'architecture des modèles sauvegardés dans des fichiers *.h5*
- New_submission: les targets prédits
- New_summarize_models: plot de la loss & mse de chaque modèle, réalisé après optimisation & validation
- New_models_history.npy: sauvegarde de l'history (loss & mse) de chaque modèle, obtenu après optimisation & validation

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

3. Utilisation

> Jupyter notebook

```
jupyter lab
```

> Programme - être dans le dossier Projet_AIOI

```
python -m aioi -a <ARG> -d <DATA>
```

avec ARG:

- opt: pour réaliser l'optimisation des réseaux de neurones
- val: pour réaliser la repeated k-fold cross-validation de chaque modèle
- app: pour réaliser l'apprentissage de chaque modèle
- eval: pour évaluer les performances de chaque modèle
- pred: pour réaliser la prédiction des targets

avec DATA - optionnel:

- classique (par défaut): réfère au dataframe du projet kaggle arn_train
- new: réfère au fichier spotrna_train_good.tsv qui contient les prédictions que nous avons réalisées des structures et des loop types

## Auteurs

BODIN Raphaël: bodin.raphael85@gmail.com

IMBERT Pierre

KLEIN Dylan: klein.dylan@outlook.com

MAURY Léo: leomaury97@gmail.com

PICHON Julien: julien.pichon@cri-paris.org

Université de Paris M2-BI

## Date

23 octobre 2020
