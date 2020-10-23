"""
Ce module permet de réaliser des graphiques.
"""

from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import sys


def summarize_learning_rate(fit_out):
    """
    Permet d'afficher la val_loss & la loss de chaque modèle.

    Et ceux pour chaque learning rate.
    """
    for models, values in fit_out.items():
        val_loss, loss = [], []
        for fit in values:
            val_loss.append(fit.history['val_loss'])
            loss.append(fit.history['loss'])

        label = ['1E-0', '1E-1', '1E-2', '1E-3', '1E-4',
                 '1E-5', '1E-6', '1E-7']
        title = ['Val loss', 'Loss']
        color = ['black', '#839192', '#BA4A00', '#F1C40F', '#1E8449',
                 '#3498DB', '#1A5276', '#8b1538']

        fig, axs = plt.subplots(2, 1, figsize=(14,12), constrained_layout=True)

        for i in range(len(values)):
            axs[0].plot(val_loss[i], color=color[i], label=label[i])
            axs[1].plot(loss[i], color=color[i], label=label[i])

        axs[0].set_title(title[0], fontsize="x-large")
        axs[1].set_title(title[1], fontsize="x-large")

        # Ajouter la légende au dessus du plot sans changer sa taille
        axs[1].legend(loc=1, bbox_to_anchor=(0.,-0.052, 1., -0.052), ncol=10,
                      borderaxespad=0., mode='expand', fontsize='large')

        fig.suptitle(models, fontsize="xx-large")

        name = f"./Models/Optimisation/optimisation-{models}.png"
        plt.savefig(name)
        plt.clf()


def summarize_neural_network(fit_out):
    """
    Permet de sauvegarder les modèles au format png.
    """
    for models, fit in fit_out.items():
        name = f"./Models/Neural_network/network-{models}.png"
        plot_model(fit[0].model, to_file=name,
                   show_shapes=True, show_layer_names=True)


def summarize_model_old(history, suptitle):
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
