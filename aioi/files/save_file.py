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

    Ces valeurs sont obtenues à la suite de repeated kfold validation.
    """
    file_ = "./Models/kfold_validation_scores.txt"
    with open(file_, "w") as filout:
        for model in scores:
            loss, mse = scores[model]['Loss'], scores[model]['Mse']

            filout.write(
                "{} - Loss: mean={:.3f}, std={:.3f} | Mse: mean={:.3f}, sd={:.3f}\n"
                .format(model.capitalize(),
                        np.mean(loss), np.std(loss), np.mean(mse), np.std(mse))
            )

