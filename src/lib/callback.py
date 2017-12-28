# -*- coding: utf-8 -*-
"""Keras callbacks script

This module initialize keras callbacks used in training via the function
define_callbacks and allow to define new ones, such as SerializeModelCheckpoint 
here.

"""

import os
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, \
    ReduceLROnPlateau,CSVLogger,TensorBoard, LearningRateScheduler

import e3_tools.log.logger as modulelogger
logger = modulelogger.get_logger(__name__, use_color=False, level=modulelogger.logging.INFO)

       
class CustomLearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, decay=0.1):
        super(CustomLearningRateScheduler, self).__init__()
        self.decay = decay

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if epoch%500==0 and epoch!=0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr*self.decay)
            logger.warning("Learning rate changed to {}".format(lr*self.decay))
    
        
def define_callbacks(ckpt_rootpath, i):
    """Define Keras callbacks for training time.

    Args:
        ckpt_rootpath (str): Training path where to save logs and tensorboard events

    Returns:
        callbacks: initialized callbacks

    """
    # Saves the model weights after each epoch
    ckpt_callback = ModelCheckpoint(os.path.join(ckpt_rootpath,'model_%d.h5' %i),
                                    monitor='val_loss',
                                    save_best_only=True,
                                    mode='min', save_weights_only=False) 
    earlystopping_callback = EarlyStopping(monitor='loss',
                                           patience=200,
                                           mode='min',
                                           verbose=1) # Early stopping
    reducelr_callback = ReduceLROnPlateau(patience=5,
                                          verbose=1) # LR decay
    lr_decay = CustomLearningRateScheduler(decay=0.1)
    Tb = TensorBoard(log_dir=ckpt_rootpath)
    csv_callback = CSVLogger(os.path.join(ckpt_rootpath,'logs.csv'),
                             append=True) # Writes a log CSV
    callbacks = [#reducelr_callback,
                 lr_decay,
                 ckpt_callback,
                 earlystopping_callback,
                 csv_callback,
                 Tb]
    return callbacks
