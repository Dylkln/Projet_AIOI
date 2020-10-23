"""
Programme pour prédire la dégradation d'ARN viraux.

Il permet:
  1. De mettre en place différente architecture de réseau de neurones
  2. L'optimisation des réseaux de neurones obtenues
  3. De vérifier les modèles créés par cross-fold validation
  4. De prédire les données à partir des modèles sélectionnés

Usage
-----
  $ python -m aioi
"""

import argparse
import warnings
warnings.filterwarnings('ignore',category=FutureWarning) # remove tensorflow warnings
import os
import sys

from aioi.data import data
from aioi.files import read_file as rf
from aioi.files import save_file as sf
from aioi.graphique import plot
from aioi.models import models as mdl


def arguments():
    """
    Détermine les arguments 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--analyse', dest='analyse', required=True,
                        choices=['o', 'v', 'a', 'p'],
                        help="""Analyse a faire: o pour optimisation des réseaux
                        v pour la cross-fold validation
                        a pour l'apprentissage
                        p pour la prédiction""")

    return parser.parse_args()


def main():
    """
    Main program function.
    """
    args = arguments()

    arn = rf.read_json('Data/train.json'), rf.read_json('Data/test.json')

    ###
    # Préparation des données x_input du réseau de neurones
    ###
    if not os.path.exists("./Data/x_train.npy"):
        x_input = ['sequence', 'structure', 'predicted_loop_type', 'SN_filter']
        x_train = data.x_input(arn[0].query('SN_filter == 1')[x_input])
        sf.save_neural_network_data("x_train", x_train)
    else:
        x_train = rf.read_npy("./Data/x_train.npy")
        print("X_train shape: {}".format(x_train.shape), end="\n\n")

    ###
    # Préparation des données y_output du réseau de neurones
    ###
    if not os.path.exists("./Data/y_train.npy"):
        y_output = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        y_train = data.y_output(arn[0].query('SN_filter == 1')[y_output])
        sf.save_neural_network_data("y_train", y_train)
    else:
        y_train = rf.read_npy("./Data/y_train.npy")
        print("Y_train shape: {}".format(y_train.shape), end="\n\n")

    if args.analyse == "o":
        print("###########")
        print("# Optimisation des réseaux de neurones!")
        print("###########", end="\n\n")

        fit_out = mdl.define_models(x_train, y_train)

        plot.summarize_learning_rate(fit_out)
        plot.summarize_neural_network(fit_out)

    elif args.analyse == "v":
        print("###########")
        print("# Cross-fold validation")
        print("###########", end="\n\n")
        scores = mdl.repeated_kfold_validation(x_train, y_train)

        sf.save_scores(scores)

    elif args.analyse == "a":
        print("###########")
        print("# Apprentissage des modèles")
        print("###########", end="\n\n")

        history, keras_models = mdl.apprentissage(x_train, y_train)

        sf.save_history(history)
        sf.save_keras_models(keras_models)


    # Rajouter prédict: https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main()
