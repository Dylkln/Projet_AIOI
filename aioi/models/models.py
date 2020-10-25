"""
Ce module permet de définir les modèles.

  - Définition des différents modèles de réseau de neurones mis en place
  - Optimisation des réseaux de neurones - learning rate
  - Validation des réseaux de neurones - k-fold cross validation
"""

import sys
import time
from keras import models
from keras import layers
from keras import optimizers
from sklearn import model_selection
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf


def model_simple(input_shape, decay):
    """
    Définition d'un 1er modèle simple.
    """
    # Convolution layers
    input_ = layers.Input(shape = input_shape)

    conv_1 = layers.Conv1D(
        filters=45, kernel_size=(3,), activation="relu", padding="same")(input_)
    conv_2 = layers.Conv1D(
        filters=35, kernel_size=(3,), activation="relu", padding="same")(conv_1)
    conv_3 = layers.Conv1D(
        filters=25, kernel_size=(3,), activation="relu", padding="same")(conv_2)
    conv_4 = layers.Conv1D(
        filters=15, kernel_size=(3,), activation="relu", padding="same")(conv_3)

    # Output layer - resize shape of the data to fit with the ouput (68,5)
    output = layers.Conv1D(
        filters=5, kernel_size=(3,), activation="relu", padding="same")(conv_4)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=0.01, decay=decay)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    return model


def resnet_block(model, ftr):
    """
    Définition d'un block resnet constitué de deux couches de convolution 1D.

    ftr: filtre
    """
    resnet = layers.Conv1D(
        filters=ftr, kernel_size=(3,), activation="relu", padding="same")(model)
    resnet = layers.Conv1D(
        filters=ftr, kernel_size=(3,), activation="relu", padding="same")(resnet)
    conc_1 = layers.Add()([model, resnet])

    return conc_1


def model_resnet(input_shape, decay, nb_resnet):
    # Convolution layers
    input_ = layers.Input(shape = input_shape)
    conv_1 = layers.Conv1D(
        filters=15, kernel_size=(3,), activation="relu", padding="same")(input_)
    conv_2 = layers.Conv1D(
        filters=15, kernel_size=(3,), activation="relu", padding="same")(conv_1)
    conv_3 = layers.Conv1D(
        filters=15, kernel_size=(3,), activation="relu", padding="same")(conv_2)

    # Resnet layers - simple
    resnet_ = conv_3
    for _ in range(nb_resnet):
        resnet_ = resnet_block(resnet_, 15)

    # Output layer - resize shape of the data to fit with the ouput (68,5)
    output = layers.Conv1D(
        filters=5, kernel_size=(3,), activation="relu", padding="same")(resnet_)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=0.01, decay=decay)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    return model


def model_resnet_10(input_shape, decay):
    """
    Modèle avec 10 block resnet.
    """
    return model_resnet(input_shape, decay, 10)


def model_compact_data(input_shape, decay):
    """
    Un modèle ou on test l'impact de compacter les données avant de les reshape.
    """
    # Convolution layers
    input_ = layers.Input(shape = input_shape)
    conv_1 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(input_)
    conv_2 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(conv_1)
    conv_3 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(conv_2)

    # Compacter les données
    compact_ = conv_3
    flt = 14
    while flt > 1:
        compact_ = layers.Conv1D(filters=flt, kernel_size=(3,),
                                 activation="relu", padding="same")(compact_)
        flt -= 3

    # Output layer - resize shape of the data to fit with the ouput (68,5)
    output = layers.Conv1D(
        filters=5, kernel_size=(3,), activation="relu", padding="same")(compact_)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=0.01, decay=decay)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    return model


def model_scatter_data(input_shape, decay):
    """
    Un modèle ou on test l'impact d'éparpiller les données avant de les reshape.
    """
    # Convolution layers
    input_ = layers.Input(shape = input_shape)
    conv_1 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(input_)
    conv_2 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(conv_1)
    conv_3 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(conv_2)

    # Éparpiller les données
    scatter_ = conv_3
    flt = 35
    while flt < 333:
        scatter_ = layers.Conv1D(filters=flt, kernel_size=(3,),
                                 activation="relu", padding="same")(scatter_)
        flt += 49

    # Output layer - resize shape of the data to fit with the ouput (68,5)
    output = layers.Conv1D(
        filters=5, kernel_size=(3,), activation="relu", padding="same")(scatter_)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=0.01, decay=decay)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    return model


def inception_block(model):
    """
    Définition d'un block inception simple.
    """
    conv_1 = layers.Conv1D(
        filters=14, kernel_size=(1,), activation="relu", padding="same")(model)
    conv_1 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(conv_1)

    conv_2 = layers.Conv1D(
        filters=14, kernel_size=(1,), activation="relu", padding="same")(model)
    conv_2 = layers.Conv1D(
        filters=14, kernel_size=(5,), activation="relu", padding="same")(conv_2)

    conv_3 = layers.Conv1D(
        filters=14, kernel_size=(1,), activation="relu", padding="same")(model)

    conc_1 = layers.Concatenate()([conv_1, conv_2, conv_3])

    return conc_1


def model_inception(input_shape, decay):
    """
    Modèle GoogLeNet - inception.
    """
    # Convolution layers
    input_ = layers.Input(shape = input_shape)
    conv_1 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(input_)
    conv_2 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(conv_1)
    conv_3 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(conv_2)

    inception_ = conv_3
    for _ in range(2):
        inception_ = inception_block(inception_)

    # Output layer - resize shape of the data to fit with the ouput (68,5)
    output = layers.Conv1D(
        filters=5, kernel_size=(3,), activation="relu", padding="same")(inception_)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=0.01, decay=decay)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    return model


def resnext_block(model, flt):
    """
    Définition d'un block resnext.
    """
    resnext = []
    for i in range(16):
        tmp = resnet_block(model, flt)
        resnext.append(tmp)
    conc_1 = layers.Concatenate()(resnext)

    return conc_1


def model_resnext(input_shape, decay):
    """
    Modèle resnext - combinaison réseaux resnet & inception.
    """
    # Convolution layers
    input_ = layers.Input(shape = input_shape)
    conv_1 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(input_)

    resnext_ = resnext_block(conv_1, 14)

    # Output layer - resize shape of the data to fit with the ouput (68,4)
    output = layers.Conv1D(
        filters=5, kernel_size=(3,), activation="relu", padding="same")(resnext_)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=0.01, decay=decay)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    return model


def define_models(x_train, y_train):
    """
    Réalise l'apprentissage de chaque modèle pour un learning rate donné.

    Cela permet de réaliser une optimisation des différents réseaux de neurones.
    """
    # Shape des data en entrée
    input_shape = (68,14)

    # Decay rates of the learning rates
    decay_rates = [1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7, 1E-8]

    # Liste des modèles utilisés
    list_model = [model_simple, model_resnet_10, model_compact_data,
                  model_scatter_data, model_inception, model_resnext]
    
    fit_out = {}

    for model in list_model:
        fit_out[model.__name__] = []
        print("\n#####\n{}\n#####".format(model.__name__))
        time.sleep(30)
        for decay in decay_rates:
            check(decay_rates.index(decay))
            time.sleep(3)
            mdl = model(input_shape, decay)
            fit = mdl.fit(x=x_train, y=y_train, epochs=100, batch_size=10,
                          verbose=1, validation_split=0.2)
            fit_out[model.__name__].append(fit)
            tf.keras.backend.clear_session()

    return fit_out


def repeated_kfold_validation(X, Y):
    """
    Réalise une k-fold Cross-validation pour évaluer les différents modèles.

    Une étape d'optimisation des modèles a été réalisée au préalable.
    """
    # Shape des data en entrée
    input_shape = (68,14)

    # Liste des modèles utilisés
    list_model = [model_simple, model_resnet_10, model_compact_data,
                  model_scatter_data, model_inception, model_resnext]

    # Learing rate optimisé pour chaque modèle
    decay_rates = [1E-4, 1E-8, 1E-6, 1E-4, 1E-4, 1E-3]

    scores = {}

    for index, model in enumerate(list_model):
        print("\n#####\n{}\n#####".format(model.__name__))
        time.sleep(30)

        scores[model.__name__] = {'Loss': [], 'Mse': []}
        # kfold = model_selection.StratifiedKFold(n_splits=10)

        for i in range(10):
            print("Kfold {} - {}".format(i+1, model.__name__))
            time.sleep(2)
            x_train, x_test, y_train, y_test = \
                model_selection.train_test_split(X, Y, test_size=0.3,
                                                 random_state=2)

            mdl = model(input_shape, decay_rates[index])
            fit = mdl.fit(x=x_train, y=y_train, epochs=100,
                          batch_size=10, verbose=1)

            # Return the loss value & metrics values
            loss, mse = mdl.evaluate(x_test, y_test, verbose=0)

            scores[model.__name__]['Loss'].append(loss)
            scores[model.__name__]['Mse'].append(mse)

    return scores


def check(index):
    print("{} / 8".format(index+1))


def apprentissage(x_train, y_train):
    """
    Apprentissage des différents réseaux de neurones après optimisation & validation.
    """
    # Shape des data en entrée
    input_shape = (68, 14)

    # Liste des modèles utilisés
    list_model = [model_simple, model_resnet_10, model_compact_data,
                  model_scatter_data, model_inception, model_resnext]

    # Learing rate optimisé pour chaque modèle
    decay_rates = [1E-4, 1E-8, 1E-6, 1E-4, 1E-4, 1E-3]

    history, models = {}, {}

    for index, model in enumerate(list_model):
        print("\n#####\n{}\n#####".format(model.__name__))
        time.sleep(30)

        mdl = model(input_shape, decay_rates[index])
        fit = mdl.fit(x=x_train, y=y_train, epochs=175, batch_size=10, verbose=1,
                      validation_split=0.2)

        history[model.__name__] = fit.history
        models[model.__name__] = mdl

    return history, models


def prediction(x_test, keras_models):
    """
    Méthode pour réaliser une prédiction à partir d'un jeu de données de test.
    """
    y_test = {}
    for mdl in keras_models:
        y_test[mdl] = keras_models[mdl].predict(x_test)
    return y_test


if __name__ == "__main__":
    sys.exit()  # Aucune action souhaitée
