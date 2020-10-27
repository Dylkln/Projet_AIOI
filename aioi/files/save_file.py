""""
Permet de sauvegarder les données.
"""

import csv
import numpy as np
import sys

def save_neural_network_data(name, data):
    """
    Permet de sauvegarder les données d'input & output traitées.

    Parameter
    ---------
    name: str
        le nom du fichier
    data: numpy array
    """
    file_ = f"./Data/{name}.npy"
    with open(file_, 'wb') as filout:
        np.save(filout, data)


def save_scores(scores):
    """
    Permet de save la valeur moyenne & la standard deviation de la loss & du mse.

    Ces valeurs sont obtenues à la suite d'une étape de repeated kfold validation.
    """
    file_ = "./Models/kfold_validation_scores.txt"
    with open(file_, "w") as filout:
        filout.write("Loss & mse obtenues après une étape de repeated kfold - 10\n")
        for model in scores:
            loss, mse = scores[model]['Loss'], scores[model]['Mse']

            filout.write(
                "{} - Loss: mean={:.3f}, std={:.3f} | Mse: mean={:.3f}, sd={:.3f}\n"
                .format(model.capitalize(),
                        np.mean(loss), np.std(loss), np.mean(mse), np.std(mse))
            )


def save_keras_models(keras_models, *args):
    """
    Save des modèles keras compilés.
    """
    for type_model, model in keras_models.items():
        if args:
            file_ = f"./Models/{args[0].capitalize()}_keras_models/{type_model}.h5"
        else:
            file_ = f"./Models/Keras_models/{type_model}.h5"
        model.save(file_)


def save_history(history, *args):
    """
    Save de fit_out.history de chaque modèle - loss, val_loss & mse, val_mse
    """
    if args:
        file_ = f"./Models/{args[0].capitalize()}_models_history.npy"
    else:
        file_ = "./Models/models_history.npy"
    np.save(file_, history)


def save_submission(output, model, *args):
    if args:
        file_ = f"Models/{args[0].capitalize()}_submission/submission-{model}.csv"
    else:
        file_ = f"Models/Submission/submission-{model}.csv"

    with open(file_, 'w') as filout:
        fields = [
            'id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C',
            'deg_50C'
        ]
        f_writer = csv.DictWriter(filout, fieldnames=fields)
        f_writer.writeheader()

        for id_seq in output:
            for i in range(len(output[id_seq]['reactivity'])):
                data = {
                    'id_seqpos': f"{id_seq}_{i}",
                    'reactivity': output[id_seq]['reactivity'][i],
                    'deg_Mg_pH10': output[id_seq]['deg_Mg_pH10'][i],
                    'deg_pH10': output[id_seq]['deg_pH10'][i],
                    'deg_Mg_50C': output[id_seq]['deg_Mg_50C'][i],
                    'deg_50C': output[id_seq]['deg_50C'][i]
                }
                f_writer.writerow(data)


if __name__ == "__main__":
    sys.exit()  # Aucune action souhaitée
