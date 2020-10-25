"""
Ce module permet de préparer les données pour le réseau de neurones.
"""

import pandas as pd
import numpy as np
import sys
from keras.utils import to_categorical


def formatage_x(arn):
    """
    Formatage des X input, i.e découpage des séquences de taille 107 & 130 en
    séquence de taille 68.

    68 correspond à la taille minimum des séquences scored et a donc été choisi.
    """
    col_names = ['id', 'sequence', 'structure', 'predicted_loop_type', 'seq_length']
    X_new = {'index': [], 'id': [], 'sequence': [], 'structure': [],
             'predicted_loop_type': [], 'seq_length': []}

    for index, row  in arn[col_names].iterrows():
        # Séquence de taille 107 - séparer la séquence [:68] & [107-68:]
        # Séquence de taille 130 - séparer la séquence [:107] & [130-107:]

        X_new['index'].append(index)
        X_new['seq_length'].append(row['seq_length'])
        for key in col_names[:-1]:
            X_new[key].append(row[key][:68])

        X_new['index'].append(index)
        X_new['seq_length'].append(row['seq_length'])
        if row['seq_length'] == 107:
            for key in col_names[:-1]:
                X_new[key].append(row[key][107-68:])
        else:
            for key in col_names[:-1]:
                X_new[key].append(row[key][130-68:])

    return pd.DataFrame.from_dict(X_new)


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
        y_train.append(tmp)

        # Les 1589 séquences du x_train sont découpées en séquences de taille 68,
        # soit - part1 [0:68]
        #      - part2 [107-68:]
        # x_train size devient donc 1589*2 = 3178
        # c'est pourquoi dédoublement dans y_train des valeurs prédites
        # ainsi y_train size = 3178
        # on utilise ce qu'on connaît (les 68 1er) pour prédire l'inconnu

    y_train = np.array(y_train)

    print("Y_train shape: {}".format(y_train.shape), end="\n\n")

    return y_train


def new_x(arn):
    """
    Formatage du data frame arn_test & de x_test au bon format.

    Return
    ------
    arn_test: pandas data frame
        les données de test au format df
    x_new: numpy array
        les données de test au bon format sur lesquel les prédiction vont être faites
    """
    arn_test = formatage_x(arn)

    xnew_seq = x_sequence(arn_test['sequence'])
    xnew_struc = x_structure(arn_test['structure'])
    xnew_loops = x_predicted_loops(arn_test['predicted_loop_type'])

    x_new = x_concatenation(xnew_seq, xnew_struc, xnew_loops)

    print("Data frame arn_test shape: {}".format(arn_test.shape))
    print("Sequences shape: {} - 4 classes".format(xnew_seq.shape))
    print("Structures shape: {} - 3 classes".format(xnew_struc.shape))
    print("Loops shape: {} - 7 classes".format(xnew_loops.shape))
    print("x_new shape: {}".format(x_new.shape), end="\n\n")

    return arn_test, x_new


def traiter_predict_output(arn, predict):
    """
    Chaque arn a été découpé en deux segements de taille 68.

    La prédiction a été ainsi réalisée:

      > ARN de taille 107

        - De la position 1 à 68
        - De la position 29 (107-68) à 107

      > ARN de taille 130

      - De la postion 1 à 68
      - De la position 62 (130-68) à 130

    Il est donc nécessaire de combiner les valeurs obtenues pour obtenir la
    prédiction sur l'ensemble de la séquence.
    """
    output = {}

    for i in range(0, len(predict), 2):
        seq_length = arn['seq_length'][i]
        id_seq = arn['id'][i]

        output[id_seq] = {
            'reactivity': [], 'deg_Mg_pH10': [], 'deg_pH10': [], 'deg_Mg_50C': [],
            'deg_50C': []
        }

        if seq_length == 107:

            # Ajout des 39 premiers éléments du segment 1
            for index, key in enumerate(output[id_seq].keys()):
                for ele in predict[i][:,index][:39]:
                    output[id_seq][key].append(ele)

            # Ajout des 29 éléments en commun entre segment 1 & 2
            # Ici une moyenne des valeurs observées à une position donnée est faite
            for index, key in enumerate(output[id_seq].keys()):
                sum_ = np.add(predict[i][:,index][39:], predict[i+1][:,index][:29])
                mean_ = np.divide(sum_, 2)
                for ele in mean_:
                    output[id_seq][key].append(ele)

            # Ajout des 39 derniers éléments du segment 2
            for index, key in enumerate(output[id_seq].keys()):
                for ele in predict[i+1][:,index][29:]:
                    output[id_seq][key].append(ele)

        else:
            # Ajout des 62 premiers éléments du segment 1
            for index, key in enumerate(output[id_seq].keys()):
                for ele in predict[i][:,index][:62]:
                    output[id_seq][key].append(ele)

            # Ajout des 6 éléments en commun entre segment 1 & 2
            # Ici une moyenne des valeurs observées à une position donnée est faite
            for index, key in enumerate(output[id_seq].keys()):
                sum_ = np.add(predict[i][:,index][62:], predict[i+1][:,index][:6])
                mean_ = np.divide(sum_, 2)
                for ele in mean_:
                    output[id_seq][key].append(ele)

            # Ajout des 62 derniers éléments du segment 2
            for index, key in enumerate(output[id_seq].keys()):
                for ele in predict[i+1][:,index][6:]:
                    output[id_seq][key].append(ele)

    return output


if __name__ == "__main__":
    sys.exit()  # Aucune action souhaitée
