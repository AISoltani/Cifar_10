# from callbacks import Step
# import callbacks

import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
import yaml
import h5py
import numpy as np

class Step(Callback):

    def __init__(self, steps, learning_rates, verbose=0):
        self.steps = steps
        self.lr = learning_rates
        self.verbose = verbose

    def change_lr(self, new_lr):
        old_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose == 1:
            print('Learning rate is %g' %new_lr)

    def on_epoch_begin(self, epoch, logs={}):
        for i, step in enumerate(self.steps):
            if epoch < step:
                self.change_lr(self.lr[i])
                return
        self.change_lr(self.lr[i+1])

    def get_config(self):
        config = {'class': type(self).__name__,
                  'steps': self.steps,
                  'learning_rates': self.lr,
                  'verbose': self.verbose}
        return config

    @classmethod
    def from_config(cls, config):
        offset = config.get('epoch_offset', 0)
        steps = [step - offset for step in config['steps']]
        return cls(steps, config['learning_rates'],
                   verbose=config.get('verbose', 0))


def onetenth_200_230(dataset, lr):
    steps = [200, 230]
    lrs = [lr, lr/10, lr/100]
    return Step(steps, lrs)


def dsn_step_200_230(dataset, lr):
    steps = [200, 230]
    lrs = [lr, lr/2.5, lr/25]
    return Step(steps, lrs)


def nin_nobn_mnist(dataset, lr):
    steps = [40, 50]
    lrs = [lr, lr/2, lr/10]
    return Step(steps, lrs)


def dsn_step_20_30(dataset, lr):
    steps = [20, 30]
    lrs = [lr, lr/2.5, lr/25]
    return Step(steps, lrs)


def dsn_step_40_60(dataset, lr):
    steps = [40, 60]
    lrs = [lr, lr/2.5, lr/25]
    return Step(steps, lrs)


def wideresnet_step(dataset, lr):
    steps = [60, 120, 160]
    lrs = [lr, lr/5, lr/25, lr/125]
    return Step(steps, lrs)
