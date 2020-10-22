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


if __name__ == "__main__":
    sys.exit()  # Aucune action souhaitée
