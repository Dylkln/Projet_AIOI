"""
Ce module permet de lire les données du projet.

Il permet de:
  - Lire les fichiers
      test.json
      train.json
      <ARNID>.npy
  - Vérifier l'existence du fichier lu
"""

import os
import pandas as pd
import numpy as np
import sys


def valide_file(fichier, path):
    """Vérifie que le fichier lu est valide.

    I.E:
      - fichier présent dans le répertoire Data/

    Parameter
    ---------
    fichier: str
        le nom du fichier
    path: str
        le path du fichier

    Return
    ------
    Boolean
      - True: fichier valide
      - False: fichier non valide
    """
    if os.path.exists(path + fichier):
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
    path = "../Data/"
    if valide_file(fichier , path):
        json_file = pd.read_json(path + fichier, lines=True)
        return json_file
    else:
        raise Exception('{} doit être présent dans le répertoire {}'
                        .format(fichier, path))


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
    path = "../Data/Files/"
    if valide_file(fichier, path):
        npy_file = np.load(path + fichier)
        return npy_file
    else:
        raise Exception('{} doit être présent dans le répertoire {}'
                        .format(fichier, path))


if __name__ == "__main__":
    sys.exit()  # Aucune action souhaitée
