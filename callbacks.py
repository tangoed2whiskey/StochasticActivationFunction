import math
import tensorflow as tf
# import tensorflow.keras.backend as K
import numpy as np

def step_decay(epoch):
    initial_lrate = 2.0
    drop = 0.998
    # drop = 0.99
    # drop = 0.95
    epochs_drop = 1
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop)) * (1.0 - momentum_schedule(epoch))
    return lrate

def resnet_lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return 1e-3#lr

def resnetv2_lr_schedule(epoch):
    initial_lrate = 0.1
    # drop = 0.998
    drop = 0.995
    # drop = 0.95
    epochs_drop = 1
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop)) * (1.0 - momentum_schedule(epoch))
    return 1.0#lrate

def momentum_schedule(epoch):
    initial_momentum = 0.5
    final_momentum   = 0.99
    epoch_max_value  = 50.0
    return 0.0
    # if epoch<epoch_max_value:
    #     return ((final_momentum - initial_momentum)/epoch_max_value) * epoch + initial_momentum
    # else:
    #     return final_momentum

class MomentumScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, verbose=0):
        super(MomentumScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self,epoch,logs=None):
        if not hasattr(self.model.optimizer,'momentum'):
            raise ValueError('Optimizer must have a "momentum" attribute')
        momentum = float(tf.keras.backend.get_value(self.model.optimizer.momentum))
        try:
            momentum = self.schedule(epoch,momentum)
        except TypeError:
            momentum = self.schedule(epoch)
        if not isinstance(momentum, (float,np.float32,np.float64)):
            raise ValueError('The output of the "schedule" function should be float')
        tf.keras.backend.set_value(self.model.optimizer.momentum,momentum)
        if self.verbose > 0:
            print('\nEpoch {}: MomentumScheduler setting momentum to {}'.format(epoch+1,momentum))
    
    def on_epoch_end(self,epoch,logs=None):
        logs = logs or {}
        logs['momentum'] = tf.keras.backend.get_value(self.model.optimizer.momentum)

class IncorrectHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.incorrect=[]
    def on_epoch_end(self,batch,logs={}):
        self.incorrect.append(1.0 - logs.get('val_acc'))

class TestAccuracy(tf.keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.incorrect=[]
    def on_epoch_end(self,batch,logs={}):
        self.incorrect.append(logs.get('val_acc'))

class IncorrectRMSEHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.incorrect=[]
    def on_epoch_end(self,batch,logs={}):
        self.incorrect.append(1.0 - logs.get('val_RMSE'))