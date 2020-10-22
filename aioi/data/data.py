"""
Ce module permet de préparer les données pour le réseau de neurones.
"""

import numpy as np
import sys
from keras.utils import to_categorical

def x_sequence(data):
    """
    Permet de réaliser le one hot encoding des séquences.

    Parameter
    ---------
    data: pandas series
        les séquences en acides aminés des arn

    Return
    ------
    x_seq: numpy array
    """
    dico_seq = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    x_seq = []

    for seq in data:
        tmp = []
        for base in seq:
            tmp.append(dico_seq[base])
        x_seq.append(to_categorical(tmp, num_classes=4, dtype=int))

    return np.array(x_seq)


def x_structure(data):
    """
    Permet de réaliser le one hot encoding des structures.

    Parameter
    ---------
    data: pandas series
        les structures des arn

    Return
    ------
    x_struc: numpy array
    """
    dico_struc = {'(': 0, ')': 1, '.': 2}
    x_struc = []

    for struc in data:
        tmp = []
        for symbol in struc:
            tmp.append(dico_struc[symbol])
        x_struc.append(to_categorical(tmp, num_classes=3, dtype=int))

    return np.array(x_struc)


def x_predicted_loops(data):
    """
    Permet de réaliser le one hot encoding des types de loop prédites.

    Parameter
    ---------
    data: pandas series
        les types de loop prédites

    Return
    ------
    x_loops: numpy array
    """
    dico_loops = {'S': 0, 'E': 1, 'B': 2, 'H': 3, 'M': 4, 'I': 5, 'X': 6}
    x_loops = []

    for loops in data:
        tmp = []
        for loop_type in loops:
            tmp.append(dico_loops[loop_type])
        x_loops.append(to_categorical(tmp, num_classes=7, dtype=int))

    return np.array(x_loops)


def x_concatenation(x_seq, x_struc, x_loops):
    """
    Regroupement des données dans une seule matrice de dimension (x, n, p).

    Avec:

      - x le nombre d'ARN, soit 1589
      - n la taille des séquences, soit 107
      - p le nombre de classes, soit 14 (4+3+7)

    Parameter
    ---------
    x_seq: numpy array
    x_struc: numpy array
    x_loops: numpy array

    Return
    ------
    data: numpy array
    """
    data = []
    for i in range(len(x_seq)):
        data.append(np.concatenate((x_seq[i], x_struc[i], x_loops[i]), axis=1))
    return np.array(data)


def x_input(arn_train):
    """
    Préparation des données d'entrée au bon format.

    Parameter
    ---------
    arn_train: pandas data frame
        les x input à traiter
    """
    print("Préparation des x_train")

    if not arn_train.shape[1] == arn.shape[1]:
        sys.exit("Error: lors de la sélection des data qui ont un un SN_filter de 1")

    x_seq = x_sequence(arn_train['sequence'])
    x_struc = x_structure(arn_train['structure'])
    x_loops = x_predicted_loops(arn_train['predicted_loop_type'])

    x_train = x_concatenation(x_seq, x_struc, x_loops)

    print("Sequences shape: {} - 4 classes".format(x_seq.shape))
    print("Structures shape: {} - 3 classes".format(x_struc.shape))
    print("Loops shape: {} - 7 classes".format(x_loops.shape))
    print("X_train shape: {}".format(x_train.shape), end="\n\n")

    return x_train


def y_output(arn_train):
    """
    Préparation des données de sortie au bon format.

    Parameter
    ---------
    arn_train: pandas data frame
        les y output à traiter

    Return
    ------
    x_loops: numpy array
    """
    y_train = []
    reactivity = arn_train['reactivity']
    deg_mg_ph = arn_train['deg_Mg_pH10']
    deg_ph = arn_train['deg_pH10']
    deg_mg_c = arn_train['deg_Mg_50C']
    deg_c = arn_train['deg_50C']

    for i in arn_train.index:
        tmp = []
        for j in range(len(deg_c[i])):
            tmp.append([
                reactivity[i][j], deg_mg_ph[i][j], deg_ph[i][j],
                deg_mg_c[i][j], deg_c[i][j]
            ])
        y_train.append(tmp)

    y_train = np.array(y_train)

    print("Y_train shape: {}".format(y_train.shape), end="\n\n")

    return y_train


if __name__ == "__main__":
    sys.exit()  # Aucune action souhaitée
