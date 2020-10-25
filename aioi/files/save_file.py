""""
Permet de sauvegarder les données.
"""

import numpy as np

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


def save_keras_models(keras_models):
    """
    Save des modèles keras compilés.
    """
    for type_model, model in keras_models.items():
        file_ = f"./Models/Keras_models/{type_model}.h5"
        model.save(file_)


def save_history(history):
    """
    Save de fit_out.history de chaque modèle - loss, val_loss & mse, val_mse
    """
    file_ = "./Models/models_history.npy"
    np.save(file_, history)
