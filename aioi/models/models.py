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


def resize_shape_data(model):
    model = layers.Conv1D(
        filters=5, kernel_size=(20,), activation="relu")(model)
    model = layers.Conv1D(
        filters=5, kernel_size=(20,), activation="relu")(model)
    model = layers.Conv1D(
        filters=5, kernel_size=(2,), activation="relu")(model)

    return model


def model_simple(input_shape, learning_rate):
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
    output = resize_shape_data(conv_4)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=learning_rate)
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


def model_resnet(input_shape, learning_rate, nb_resnet):
    # Convolution layers
    input_ = layers.Input(shape = input_shape)
    conv_1 = layers.Conv1D(
        filters=30, kernel_size=(3,), activation="relu", padding="same")(input_)
    conv_2 = layers.Conv1D(
        filters=30, kernel_size=(3,), activation="relu", padding="same")(conv_1)
    conv_3 = layers.Conv1D(
        filters=30, kernel_size=(3,), activation="relu", padding="same")(conv_2)

    # Resnet layers - simple
    resnet_ = conv_3
    for _ in range(nb_resnet):
        resnet_ = resnet_block(resnet_, 30)

    # Output layer - resize shape of the data to fit with the ouput (68,5)
    output = resize_shape_data(resnet_)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=learning_rate)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    return model


def model_resnet_10(input_shape, learning_rate):
    """
    Modèle avec 10 block resnet.
    """
    return model_resnet(input_shape, learning_rate, 10)


def model_compact_data(input_shape, learning_rate):
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
    output = resize_shape_data(compact_)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=learning_rate)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    return model


def model_scatter_data(input_shape, learning_rate):
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
    while flt < 400:
        scatter_ = layers.Conv1D(filters=flt, kernel_size=(3,),
                                 activation="relu", padding="same")(scatter_)
        flt += 49

    # Output layer - resize shape of the data to fit with the ouput (68,5)
    output = resize_shape_data(scatter_)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=learning_rate)
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


def model_inception(input_shape, learning_rate):
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
    output = resize_shape_data(inception_)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=learning_rate)
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


def model_resnext(input_shape, learning_rate):
    """
    Modèle resnext - combinaison réseaux resnet & inception.
    """
    # Convolution layers
    input_ = layers.Input(shape = input_shape)
    conv_1 = layers.Conv1D(
        filters=14, kernel_size=(3,), activation="relu", padding="same")(input_)

    resnext_ = resnext_block(conv_1, 14)

    # Output layer - resize shape of the data to fit with the ouput (68,4)
    output = resize_shape_data(resnext_)
    model = models.Model(inputs=input_, outputs=output)

    # Compilation
    opt = optimizers.SGD(lr=learning_rate)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    return model


def define_models(x_train, y_train):
    """
    Réalise l'apprentissage de chaque modèle pour un learning rate donné.

    Cela permet de réaliser une optimisation des différents réseaux de neurones.
    """
    # Shape des data en entrée
    input_shape = (107,14)

    # Learning rate à appliquer à chaque modèle
    learning_rates = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4,
                      1E-5, 1E-6, 1E-7]

    # Liste des modèles utilisés
    list_model = [model_simple, model_resnet_10, model_compact_data,
                  model_scatter_data, model_inception, model_resnext]

    fit_out = {}

    for model in list_model:
        fit_out[model.__name__] = []
        print("\n#####\n{}\n#####".format(model.__name__))
        time.sleep(30)
        for lr in learning_rates:
            check(learning_rates.index(lr))
            time.sleep(3)
            mdl = model(input_shape, lr)
            fit = mdl.fit(x=x_train, y=y_train, epochs=75, batch_size=10,
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
    input_shape = (107,14)

    # Liste des modèles utilisés
    list_model = [model_simple, model_resnet_10, model_compact_data,
                  model_scatter_data, model_inception, model_resnext]

    # Learing rate optimisé pour chaque modèle
    learning_rate = [1E-2, 1E-6, 1E-1, 1E-2, 1E-2, 1E-2]

    scores = {}

    for index, model in enumerate(list_model):
        print("\n#####\n{}\n#####".format(model.__name__))
        #time.sleep(30)

        scores[model.__name__] = {'Loss': [], 'Mse': []}
        # kfold = model_selection.StratifiedKFold(n_splits=10)

        for i in range(10):
            print("Kfold {}".format(i+1))
            time.sleep(2)
            x_train, x_test, y_train, y_test = \
                model_selection.train_test_split(X, Y, test_size=0.3,
                                                 random_state=2)

            mdl = model(input_shape, learning_rate[index])
            fit = mdl.fit(x=x_train, y=y_train, epochs=75,
                          batch_size=10, verbose=0, validation_split=0.2)

            # Return the loss value & metrics values
            loss, mse = mdl.evaluate(x_test, y_test, verbose=0)

            scores[model.__name__]['Loss'].append(loss)
            scores[model.__name__]['Mse'].append(mse)

    return scores


def check(index):
    print("{} / 8".format(index+1))


if __name__ == "__main__":
    sys.exit()  # Aucune action souhaitée
