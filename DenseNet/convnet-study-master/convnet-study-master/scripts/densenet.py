import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, merge, Activation, Dropout, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# from rme.datasets import cifar10
from keras.datasets import cifar10

from rme.callbacks import Step, MetaCheckpoint

import os
import yaml
import numpy as np


def add_layer(x, nb_channels, kernel_size=3, dropout=0., l2_reg=1e-4):
    out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(x)
    out = Activation('relu')(out)
    out = Convolution2D(nb_channels, kernel_size, kernel_size,
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    if dropout > 0:
        out = Dropout(dropout)(out)
    return out


def dense_block(x, nb_layers, growth_rate, dropout=0., l2_reg=1e-4):
    for i in range(nb_layers):
        # Get layer output
        out = add_layer(x, growth_rate, dropout=dropout, l2_reg=l2_reg)
        if K.image_dim_ordering() == 'tf':
            merge_axis = -1
        elif K.image_dim_ordering() == 'th':
            merge_axis = 1
        else:
            raise Exception('Invalid dim_ordering: ' + K.image_dim_ordering())
        # Concatenate input with layer ouput
        # x = merge([x, out], mode='concat', concat_axis=merge_axis)
        x = keras.layers.Concatenate(axis=merge_axis)([x, out])

    return x


def transition_block(x, nb_channels, dropout=0., l2_reg=1e-4):
    x = add_layer(x, nb_channels, kernel_size=1,
                  dropout=dropout, l2_reg=l2_reg)
    # x = Convolution2D(n_channels, 1, 1, border_mode='same',
    #                   init='he_normal', W_regularizer=l2(l2_reg))(x)
    x = AveragePooling2D()(x)
    return x


def densenet_model(nb_blocks, nb_layers, growth_rate, dropout=0., l2_reg=1e-4,
                   init_channels=16):
    n_channels = init_channels
    inputs = Input(shape=(32, 32, 3))
    x = Convolution2D(init_channels, 3, 3, border_mode='same',
                      init='he_normal', W_regularizer=l2(l2_reg),
                      bias=False)(inputs)
    for i in range(nb_blocks - 1):
        # Create a dense block
        x = dense_block(x, nb_layers, growth_rate,
                        dropout=dropout, l2_reg=l2_reg)
        # Update the number of channels
        n_channels += nb_layers*growth_rate
        # Transition layer
        x = transition_block(x, n_channels, dropout=dropout, l2_reg=l2_reg)

    # Add last dense_block
    x = dense_block(x, nb_layers, growth_rate, dropout=dropout, l2_reg=l2_reg)
    # Add final BN-Relu
    x = BatchNormalization(gamma_regularizer=l2(l2_reg),
                           beta_regularizer=l2(l2_reg))(x)
    x = Activation('relu')(x)
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, W_regularizer=l2(l2_reg))(x)
    x = Activation('softmax')(x)

    model = Model(input=inputs, output=x)
    return model

# Apply preprocessing as described in the paper: normalize each channel
# individually. We use the values from fb.resnet.torch, but computing the values
# gets a very close answer.


def preprocess_data(data_set):
    mean = np.array([125.3, 123.0, 113.9])
    std = np.array([63.0, 62.1, 66.7])

    data_set = np.array(data_set, dtype='float64')

    data_set -= mean
    data_set /= std
    return data_set


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    from sklearn.metrics import confusion_matrix

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# if __name__ == '__main__':
augmented = True

if augmented:
    file_name = 'densenet_augmented'
    dropout = 0.
    print('Using augmented dataset.')
else:
    file_name = 'densenet'
    dropout = 0.2
    print('Using regular dataset.')

save_path = os.path.join('weights', 'cifar10', file_name)

# Define optimizer
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

model = densenet_model(3, 12, 12, dropout=dropout)
model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Model compiled.')

print('Loading data.')
# train_set, _, test_set = cifar10.load('data/cifar10', valid_ratio=0.,
#                                         gcn=False, zca=False)
# train_set['data'] = preprocess_data(train_set['data'])
# test_set['data'] = preprocess_data(test_set['data'])

# train_set, _, test_set = cifar10.load('data/cifar10', valid_ratio=0.,gcn=False, zca=False)
(train_set, y_train), (test_set, y_test) = cifar10.load_data()
train_set = preprocess_data(train_set)
test_set = preprocess_data(test_set)
print('Data loaded.')

# Train for 300 epochs as in the paper
nb_epoch = 300
# nb_epoch = 1
offset = 0
# Define callbacks for checkpointing and scheduling
callbacks = []
# callbacks.append(ModelCheckpoint(save_path + '.h5'))
steps = [nb_epoch/2 - offset, 3*nb_epoch/4 - offset]
schedule = Step(steps, [0.1, 0.01, 0.001], verbose=1)
callbacks.append(schedule)

# Custom meta checkpoint that saves training information at each epoch.
# callbacks.append(MetaCheckpoint(save_path + '.meta'))
# callbacks.append(MetaCheckpoint(save_path + '.meta', schedule=schedule))

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
if augmented:
    data_gen = ImageDataGenerator(horizontal_flip=True,
                                  width_shift_range=0.125,
                                  height_shift_range=0.125,
                                  fill_mode='constant')
    # data_iter = data_gen.flow(train_set['data'], train_set['labels'],batch_size=64, shuffle=True)

    data_iter = data_gen.flow(train_set, y_train, batch_size=64, shuffle=True)

    # y_test = y_test.argmax(axis=1)
    # y_test=

    model.fit_generator(data_iter,
                        samples_per_epoch=train_set.shape[0],
                        nb_epoch=(nb_epoch - offset),
                        verbose=1,
                        validation_data=(test_set,
                                         y_test),
                        callbacks=callbacks)
else:
    model.fit(train_set, y_train, batch_size=64,
              nb_epoch=(nb_epoch - offset), verbose=1,
              validation_data=(test_set, y_test),
              callbacks=callbacks, shuffle=True)


y_prob = model.predict(test_set)
y_pred = y_prob.argmax(axis=-1)

y_test = np.argmax(y_test, axis=1)
class_name = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

plot_confusion_matrix(y_test, y_pred, classes=class_name,
                      normalize=False, title='DenseNet using data augmentation')
print(
    f'DenseNet using data augmentation: {accuracy_score(y_test, y_pred)}')
plt.show()
