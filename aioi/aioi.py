"""
Programme pour prédire la dégradation d'ARN viraux.

Il permet:
  1. De mettre en place différente architecture de réseau de neurones
  2. L'optimisation des réseaux de neurones obtenues
  3. De vérifier les modèles créés par cross-fold validation
  4. De prédire les données à partir des modèles sélectionnés

Usage
-----
  $ python -m aioi -a ARG
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
    Détermine les arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--analyse', dest='analyse', required=True,
                        choices=['opt', 'val', 'app', 'eval', 'pred'],
                        help="""Analyse a faire: opt pour optimisation des réseaux
                        val pour la cross-fold validation
                        app pour l'apprentissage
                        eval pour évaluer les performances des modèles
                        pred pour la prédiction""")
    parser.add_argument('-d', '--data', dest='data', default="classique",
                        choices=['classique', 'new'],
                        help="""Le fichier de train à utiliser pour l'apprentissage -
                         classique réfère au dataframe du projet kaggle arn_train -
                         new réfère au fichier spotrna_train_good.tsv qui contient
                        les prédictions que nous avons réalisées des structures et
                        des loop types
                        """)

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
        arn_train = data.formatage_x(arn[0].query('SN_filter == 1'))
        x_train = data.x_input(arn_train)
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

    ###
    # Modèle keras
    ###
    if args.analyse == "opt":
        print("###########")
        print("# Optimisation des réseaux de neurones!")
        print("###########", end="\n\n")

        fit_out = mdl.define_models(x_train, y_train)

        plot.summarize_learning_rate(fit_out)
        plot.summarize_neural_network(fit_out)

        print("\nOVER\n")

    elif args.analyse == "val":
        print("###########")
        print("# Cross-fold validation")
        print("###########", end="\n\n")
        scores = mdl.repeated_kfold_validation(x_train, y_train)

        sf.save_scores(scores)

        print("\nOVER\n")

    elif args.analyse == "app":
        print("###########")
        print("# Apprentissage des modèles")
        print("###########", end="\n\n")

        if args.data == "new":
            # Data sélectionné pour l'apprentissage: spotrna_train_good.tsv
            # Mis à jour de arn_train
            new_arn = rf.read_tsv(arn[0].query('SN_filter == 1'))
            arn_train = data.formatage_x(new_arn)
            x_train = data.x_input(arn_train)

        sys.exit()
        history, keras_models = mdl.apprentissage(x_train, y_train)

        if args.data == "new":
            sf.save_history(history, "new")
            sf.save_keras_models(keras_models, "new")
        sf.save_history(history)
        sf.save_keras_models(keras_models)

        print("\nOVER\n")

    elif args.analyse == "eval":
        print("###########")
        print("# Evaluation des performances du modèles")
        print("###########", end="\n\n")

        # Analyse performance des modèles en apprentissage & test
        if args.data == "new":
            history = rf.load_history("new")
            plot.summarize_models(history, "new")
        else:
            history = rf.load_history()
            plot.summarize_models(history)

    elif args.analyse == "pred":
        print("###########")
        print("# Prédiction")
        print("###########", end="\n\n")

        if args.data == "new":
            keras_models = rf.load_keras_models("new")
        else:
            keras_models = rf.load_keras_models()

        arn_test, x_test = data.new_x(arn[1])

        predict = mdl.prediction(x_test, keras_models)

        for model in predict:
            output = data.traiter_predict_output(arn_test, predict[model])

            if args.data == "new":
                sf.save_submission(output, model, "new")
            else:
                sf.save_submission(output, model)

        print("\nOVER\n")


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main()
