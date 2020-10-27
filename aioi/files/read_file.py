"""
Ce module permet de lire les données du projet.

Il permet de:
  - Lire les fichiers
      test.json
      train.json
      <ARNID>.npy
  - Vérifier l'existence du fichier lu
"""

import csv
import os
import pandas as pd
import numpy as np
import sys

from keras import models


def valide_file(fichier):
    """Vérifie que le fichier lu est valide.

    I.E:
      - fichier présent dans le répertoire Data/

    Parameter
    ---------
    fichier: str
        le nom du fichier

    Return
    ------
    Boolean
      - True: fichier valide
      - False: fichier non valide
    """
    if os.path.exists(fichier):
        return True
    return False


def read_json(fichier):
    """
    Méthode de lecture d'un fichier .json.

    Parameter
    ---------
    fichier: str
        le nom du fichier

    Return
    ------
    json_file: pandas data frame
        les données
    """
    if valide_file(fichier):
        json_file = pd.read_json(fichier, lines=True)
        return json_file
    else:
        raise Exception('{} est absent'.format(fichier))


def read_npy(fichier):
    """
    Méthode de lecture d'un fichier .npy.

    Parameter
    ---------
    fichier: str
        le nom du fichier

    Return
    ------
    npy_file: numpy array
        les données
    """
    if valide_file(fichier):
        npy_file = np.load(fichier)
        return npy_file


def load_keras_models(*args):
    """
    Load les modèles stockés dans des fichier .h5.

    Return
    ------
    keras_models: dictionary
      -key: modèle type
      -value: keras model
    """
    keras_models = {}

    if args:
        path = f"./Models/{args[0].capitalize()}_keras_models/"
    else:
        path = "./Models/Keras_models/"

    for mdl in os.listdir(path):
        file_ = path + mdl
        name_ = mdl.split('.')[0]
        keras_models[name_] = models.load_model(file_)

    return keras_models


def load_history(*args):
    if args:
        file_ = f"./Models/{args[0].capitalize()}_models_history.npy"
    else:
        file_ = "./Models/models_history.npy"
    return np.load(file_, allow_pickle=True).item()


def read_tsv(arn):
    """
    Lis le fichier ./Data/spotrna_train.tsv
    """
    with open("./Data/spotrna_train_good.tsv") as filin:
        f_reader = csv.DictReader(filin, delimiter="\t")
        data_ = {'structure': [], 'predicted_loop_type': []}

        for row in f_reader:
            data_['structure'].append(row['structure'])
            data_['predicted_loop_type'].append(row['predicted_loop_type'])

    data_ = pd.DataFrame(data_)

    arn.update(data_)

    return arn


if __name__ == "__main__":
    sys.exit()  # Aucune action souhaitée
