"""
Ce module permet de présenter les résultats une fois que le modèle a été évalué.

Il est représenté l'accuracy et la loss d'un modèle.

  - La ligne bleu renseigne des performances en resubstitution (app)
  - La ligne orange renseigne des performances en généralisation (test)
"""

from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import sys


def summarize_model(history, suptitle):
    """
    Permet de faire les deux graphes.

    Parameter
    ---------
    history
        ce qui est retourné par le fit d'un modèle
    suptitle: str
        le titre du graphe
    """
    label = [('loss', 'val_loss'), ('accuracy', 'val_accuracy')]
    title = ['Cross entropy loss', 'Classification accuracy']

    fig, axs = plt.subplots(2, 1, figsize=(9,6), constrained_layout=True)

    for i in range(2):
        axs[i].plot(history[label[i][0]], color='blue', label=label[i][0])
        axs[i].plot(history[label[i][1]], color='orange', label=label[i][1])
        axs[i].set_title(title[i], fontsize="medium")
        
        # Ajouter la légende au dessus du plot sans changer sa taille
        axs[i].legend(loc=3, bbox_to_anchor=(0., 1.02, 1., .102), ncol=2, 
                      borderaxespad=0., mode='expand', fontsize='medium')

    title = "./Fig/" + suptitle
    fig.suptitle(title, fontsize="x-large")

    plt.show()
    plt.clf()
    
    
if __name__ == "__main__":
    sys.exit()  # Aucune action souhaitée