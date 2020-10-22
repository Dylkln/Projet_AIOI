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
    file_ = "./Data/" + name + ".npy"
    with open(file_, 'wb') as filout:
        np.save(filout, data)
