#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds


ex = Experiment("MNIST-Convnet")
ex.observers.append(MongoObserver.create())
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def confnet_config():
    batch_size = 128
    epochs = 12
    convolution_layers = [
        {'kernels': 32, 'size': (3, 3), 'activation': 'relu'},
        {'kernels': 64, 'size': (3, 3), 'activation': 'relu'}
    ]
    maxpooling_pool_size = (2, 2)
    maxpooling_dropout = 0.25
    dense_layers = [
        {'size': 128, 'activation': 'relu'}
    ]
    dense_dropout = 0.0
    final_dropout = 0.5


@ex.capture
def log_performance(_run, logs):
    _run.add_artifact("weights.hdf5")
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("accuracy", float(logs.get('accuracy')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.log_scalar("val_accuracy", float(logs.get('val_accuracy')))
    _run.result = float(logs.get('val_accuracy'))


@ex.automain
def define_and_train(batch_size, epochs,
                     convolution_layers,
                     maxpooling_pool_size, maxpooling_dropout,
                     dense_layers, dense_dropout,
                     final_dropout):
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    from keras.utils import to_categorical
    from keras.losses import categorical_crossentropy
    from keras.optimizers import Adadelta
    from keras import backend as K
    from keras.callbacks import ModelCheckpoint, Callback

    class LogPerformance(Callback):
        def on_epoch_end(self, _, logs={}):
            log_performance(logs=logs)

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    input_shape = (1, 28, 28) if K.image_data_format() == 'channels_first' else (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], *input_shape).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], *input_shape).astype('float32') / 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(convolution_layers[0]['kernels'],
                     kernel_size=convolution_layers[0]['size'],
                     activation=convolution_layers[0]['activation'],
                     input_shape=input_shape))
    for layer in convolution_layers[1:]:
        model.add(Conv2D(layer['kernels'],
                         kernel_size=layer['size'],
                         activation=layer['activation']))
    model.add(MaxPooling2D(pool_size=maxpooling_pool_size))
    model.add(Dropout(maxpooling_dropout))
    model.add(Flatten())
    for layer in dense_layers:
        model.add(Dense(layer['size'], activation=layer['activation']))
        if layer != dense_layers[-1]:
            model.add(Dropout(dense_dropout))
    model.add(Dropout(final_dropout))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[ModelCheckpoint("weights.hdf5", monitor='val_loss',
                                         save_best_only=True, mode='auto', period=1),
                         LogPerformance()])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score[1]
